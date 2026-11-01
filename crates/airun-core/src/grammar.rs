//! Constrained generation: GBNF grammars and JSON schemas, as llama.cpp's
//! `grammar`, `json_schema` and `response_format` take them.
//!
//! A grammar is parsed into rules of alternatives of sequences (groups and
//! repetitions become generated rules, as llama.cpp does) and matched by a
//! pushdown automaton over Unicode code points: the state is a set of stacks,
//! each topped by the character class it needs next. A token is allowed when
//! its text, appended to what was generated, keeps some stack alive; a
//! character split across tokens is carried to the next token. Generation may
//! end when some stack is empty.
//!
//! A grammar can wait for a trigger ([`Grammar::after`]): a reply that opens
//! with reasoning thinks freely, and the constraint starts once `</think>`
//! has gone by (whitespace after it passes too).
//!
//! [`Grammar::mask`] runs once per sampled token. Checking every token of a
//! 248k-word vocabulary each step would cost tens of milliseconds, so the most
//! likely 256 are tried first and the rest only when none of them fits.

use std::collections::HashSet;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
enum Elem {
    /// A code point in one of the ranges (inclusive), or outside all of them.
    Char { ranges: Vec<(u32, u32)>, negated: bool },
    Rule(usize),
}

impl Elem {
    fn matches(&self, c: u32) -> bool {
        match self {
            Elem::Char { ranges, negated } => ranges.iter().any(|&(a, b)| a <= c && c <= b) != *negated,
            Elem::Rule(_) => false,
        }
    }
}

/// Rules, each a list of alternatives, each a sequence of elements.
#[derive(Debug, Clone)]
struct Rules {
    rules: Vec<Vec<Vec<Elem>>>,
    names: Vec<String>,
}

/// A position: rule, alternative, element index.
type Pos = (u32, u32, u32);
type Stack = Vec<Pos>;

/// Grammar and match state.
#[derive(Clone)]
pub struct Grammar {
    rules: Arc<Rules>,
    stacks: Vec<Stack>,
    /// Bytes of a character the last token left unfinished.
    partial: Vec<u8>,
    /// Each token's bytes, indexed by id, and which ids end generation.
    vocab: Arc<Vec<Vec<u8>>>,
    eog: Arc<Vec<u32>>,
    /// Text to wait for before constraining anything, and what came so far.
    trigger: Option<(Vec<u8>, Vec<u8>)>,
    /// Just triggered: whitespace before the grammar's first character passes.
    lead_ws: bool,
}

// --- parsing ------------------------------------------------------------------

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    rules: Vec<Vec<Vec<Elem>>>,
    names: Vec<String>,
    defined: Vec<bool>,
}

impl<'a> Parser<'a> {
    fn err<T>(&self, msg: &str) -> Result<T, String> {
        let line = self.src[..self.pos.min(self.src.len())].iter().filter(|&&b| b == b'\n').count() + 1;
        Err(format!("grammar line {line}: {msg}"))
    }

    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    /// Skips spaces and comments; newlines too when `newlines`.
    fn space(&mut self, newlines: bool) {
        while let Some(c) = self.peek() {
            match c {
                b' ' | b'\t' => self.pos += 1,
                b'\r' | b'\n' if newlines => self.pos += 1,
                b'#' => {
                    while self.peek().is_some_and(|c| c != b'\n') {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn name(&mut self) -> Option<String> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_') {
            self.pos += 1;
        }
        (self.pos > start).then(|| String::from_utf8_lossy(&self.src[start..self.pos]).into_owned())
    }

    fn rule_id(&mut self, name: &str) -> usize {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            return i;
        }
        self.names.push(name.to_string());
        self.rules.push(Vec::new());
        self.defined.push(false);
        self.names.len() - 1
    }

    fn fresh_rule(&mut self, base: &str) -> usize {
        let name = format!("{base}_{}", self.names.len());
        let id = self.rule_id(&name);
        self.defined[id] = true;
        id
    }

    /// One character of a literal or class, with escapes.
    fn char(&mut self) -> Result<u32, String> {
        let c = self.peek().ok_or("unexpected end of grammar")?;
        if c == b'\\' {
            self.pos += 1;
            let e = self.peek().ok_or("unexpected end after \\")?;
            self.pos += 1;
            let hex = |p: &mut Self, n: usize| -> Result<u32, String> {
                let s = std::str::from_utf8(p.src.get(p.pos..p.pos + n).ok_or("short escape")?).map_err(|e| e.to_string())?;
                p.pos += n;
                u32::from_str_radix(s, 16).map_err(|e| e.to_string())
            };
            return Ok(match e {
                b'n' => '\n' as u32,
                b't' => '\t' as u32,
                b'r' => '\r' as u32,
                b'x' => hex(self, 2)?,
                b'u' => hex(self, 4)?,
                b'U' => hex(self, 8)?,
                other => other as u32,
            });
        }
        // A UTF-8 sequence in the grammar text.
        let len = match c { 0..=0x7F => 1, 0xC0..=0xDF => 2, 0xE0..=0xEF => 3, _ => 4 };
        let s = std::str::from_utf8(self.src.get(self.pos..self.pos + len).ok_or("bad UTF-8")?)
            .map_err(|e| e.to_string())?;
        self.pos += len;
        Ok(s.chars().next().unwrap() as u32)
    }

    fn alternates(&mut self, rule_name: &str, nested: bool) -> Result<Vec<Vec<Elem>>, String> {
        let mut alts = vec![self.sequence(rule_name, nested)?];
        while self.peek() == Some(b'|') {
            self.pos += 1;
            self.space(true);
            alts.push(self.sequence(rule_name, nested)?);
        }
        Ok(alts)
    }

    fn sequence(&mut self, rule_name: &str, nested: bool) -> Result<Vec<Elem>, String> {
        let mut seq: Vec<Elem> = Vec::new();
        // Where the last element (for a repetition suffix) starts.
        let mut last_start = 0;
        loop {
            let Some(c) = self.peek() else { break };
            let start = seq.len();
            match c {
                b'"' => {
                    self.pos += 1;
                    while self.peek() != Some(b'"') {
                        if self.peek().is_none() {
                            return self.err("unterminated string");
                        }
                        let ch = self.char()?;
                        seq.push(Elem::Char { ranges: vec![(ch, ch)], negated: false });
                    }
                    self.pos += 1;
                }
                b'[' => {
                    self.pos += 1;
                    let negated = self.peek() == Some(b'^');
                    if negated {
                        self.pos += 1;
                    }
                    let mut ranges = Vec::new();
                    while self.peek() != Some(b']') {
                        if self.peek().is_none() {
                            return self.err("unterminated character class");
                        }
                        let a = self.char()?;
                        let b = if self.peek() == Some(b'-') && self.src.get(self.pos + 1) != Some(&b']') {
                            self.pos += 1;
                            self.char()?
                        } else {
                            a
                        };
                        ranges.push((a, b));
                    }
                    self.pos += 1;
                    seq.push(Elem::Char { ranges, negated });
                }
                b'.' => {
                    self.pos += 1;
                    seq.push(Elem::Char { ranges: vec![], negated: true });
                }
                b'(' => {
                    self.pos += 1;
                    self.space(true);
                    let alts = self.alternates(rule_name, true)?;
                    if self.peek() != Some(b')') {
                        return self.err("expected )");
                    }
                    self.pos += 1;
                    let id = self.fresh_rule(rule_name);
                    self.rules[id] = alts;
                    seq.push(Elem::Rule(id));
                }
                c if c.is_ascii_alphanumeric() || c == b'-' || c == b'_' => {
                    let name = self.name().unwrap();
                    let id = self.rule_id(&name);
                    seq.push(Elem::Rule(id));
                }
                b'*' | b'+' | b'?' | b'{' => {
                    if seq.is_empty() {
                        return self.err("repetition without anything to repeat");
                    }
                    let (min, max) = match c {
                        b'*' => { self.pos += 1; (0, None) }
                        b'+' => { self.pos += 1; (1, None) }
                        b'?' => { self.pos += 1; (0, Some(1)) }
                        _ => {
                            self.pos += 1;
                            self.space(false);
                            let num = |p: &mut Self| -> Option<usize> {
                                let s = p.pos;
                                while p.peek().is_some_and(|c| c.is_ascii_digit()) { p.pos += 1; }
                                std::str::from_utf8(&p.src[s..p.pos]).ok()?.parse().ok()
                            };
                            let min = num(self).ok_or("expected a number in {}")?;
                            self.space(false);
                            let max = if self.peek() == Some(b',') {
                                self.pos += 1;
                                self.space(false);
                                num(self)
                            } else {
                                Some(min)
                            };
                            self.space(false);
                            if self.peek() != Some(b'}') {
                                return self.err("expected }");
                            }
                            self.pos += 1;
                            (min, max)
                        }
                    };
                    let item: Vec<Elem> = seq.drain(last_start..).collect();
                    let rep = self.repeat(rule_name, item, min, max);
                    seq.extend(rep);
                    self.space(nested);
                    continue;
                }
                _ => break,
            }
            last_start = start;
            self.space(nested);
        }
        Ok(seq)
    }

    /// `item` repeated min..=max times (None: unbounded), as generated rules.
    fn repeat(&mut self, base: &str, item: Vec<Elem>, min: usize, max: Option<usize>) -> Vec<Elem> {
        let mut out = Vec::new();
        for _ in 0..min {
            out.extend(item.iter().cloned());
        }
        match max {
            None => {
                // R ::= item R | (empty)
                let r = self.fresh_rule(base);
                let mut rec = item.clone();
                rec.push(Elem::Rule(r));
                self.rules[r] = vec![rec, vec![]];
                out.push(Elem::Rule(r));
            }
            Some(max) if max > min => {
                // Nested optionals: (item (item (...)?)?)?
                let mut inner: Option<usize> = None;
                for _ in min..max {
                    let r = self.fresh_rule(base);
                    let mut seq = item.clone();
                    if let Some(i) = inner {
                        seq.push(Elem::Rule(i));
                    }
                    self.rules[r] = vec![seq, vec![]];
                    inner = Some(r);
                }
                out.push(Elem::Rule(inner.unwrap()));
            }
            _ => {}
        }
        out
    }

    fn parse(mut self) -> Result<Rules, String> {
        self.space(true);
        while self.peek().is_some() {
            let name = match self.name() {
                Some(n) => n,
                None => return self.err("expected a rule name"),
            };
            self.space(false);
            if !self.src[self.pos..].starts_with(b"::=") {
                return self.err("expected ::=");
            }
            self.pos += 3;
            self.space(true);
            let id = self.rule_id(&name);
            let alts = self.alternates(&name, false)?;
            self.rules[id] = alts;
            self.defined[id] = true;
            self.space(true);
        }
        if let Some(i) = self.defined.iter().position(|d| !d) {
            return Err(format!("grammar uses rule '{}' but never defines it", self.names[i]));
        }
        if !self.names.iter().any(|n| n == "root") {
            return Err("grammar has no 'root' rule".into());
        }
        Ok(Rules { rules: self.rules, names: self.names })
    }
}

// --- matching -------------------------------------------------------------------

impl Rules {
    fn elem(&self, p: Pos) -> Option<&Elem> {
        self.rules[p.0 as usize][p.1 as usize].get(p.2 as usize)
    }

    /// Expands `stack` until its top is a character element (or it is empty,
    /// which accepts), pushing every result into `out`.
    fn expand(&self, mut stack: Stack, out: &mut Vec<Stack>, seen: &mut HashSet<Stack>, depth: usize) {
        if depth > 64 {
            return; // left recursion or a runaway grammar: drop this path
        }
        loop {
            let Some(&top) = stack.last() else {
                if seen.insert(stack.clone()) {
                    out.push(stack);
                }
                return;
            };
            match self.elem(top) {
                None => {
                    stack.pop();
                }
                Some(Elem::Char { .. }) => {
                    if seen.insert(stack.clone()) {
                        out.push(stack);
                    }
                    return;
                }
                Some(Elem::Rule(r)) => {
                    let r = *r;
                    stack.pop();
                    stack.push((top.0, top.1, top.2 + 1));
                    for alt in 0..self.rules[r].len() {
                        let mut s = stack.clone();
                        s.push((r as u32, alt as u32, 0));
                        self.expand(s, out, seen, depth + 1);
                    }
                    return;
                }
            }
        }
    }

    fn start(&self) -> Vec<Stack> {
        let root = self.names.iter().position(|n| n == "root").expect("checked at parse");
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for alt in 0..self.rules[root].len() {
            self.expand(vec![(root as u32, alt as u32, 0)], &mut out, &mut seen, 0);
        }
        out
    }

    /// The stacks after consuming character `c`.
    fn advance(&self, stacks: &[Stack], c: u32) -> Vec<Stack> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for st in stacks {
            let Some(&top) = st.last() else { continue };
            if self.elem(top).is_some_and(|e| e.matches(c)) {
                let mut s = st.clone();
                s.pop();
                s.push((top.0, top.1, top.2 + 1));
                self.expand(s, &mut out, &mut seen, 0);
            }
        }
        out
    }

    /// Whether some stack's next character could start a multi-byte one.
    fn takes_non_ascii(&self, stacks: &[Stack]) -> bool {
        stacks.iter().any(|st| st.last().and_then(|&t| self.elem(t)).is_some_and(|e| match e {
            Elem::Char { ranges, negated } => *negated || ranges.iter().any(|&(_, b)| b >= 0x80),
            _ => false,
        }))
    }
}

/// Decodes `bytes` into code points; the trailing bytes of an unfinished
/// character are returned apart. None on invalid UTF-8.
fn decode(bytes: &[u8]) -> Option<(Vec<u32>, Vec<u8>)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        let len = match b { 0..=0x7F => 1, 0xC0..=0xDF => 2, 0xE0..=0xEF => 3, 0xF0..=0xF7 => 4, _ => return None };
        if i + len > bytes.len() {
            return Some((out, bytes[i..].to_vec()));
        }
        let s = std::str::from_utf8(&bytes[i..i + len]).ok()?;
        out.push(s.chars().next()? as u32);
        i += len;
    }
    Some((out, Vec::new()))
}

impl Grammar {
    /// Parses a GBNF grammar. `vocab` holds each token's bytes, `eog` the
    /// tokens that end generation.
    pub fn new(gbnf: &str, vocab: Arc<Vec<Vec<u8>>>, eog: Arc<Vec<u32>>) -> Result<Self, String> {
        let rules = Parser { src: gbnf.as_bytes(), pos: 0, rules: Vec::new(), names: Vec::new(), defined: Vec::new() }
            .parse()?;
        let stacks = rules.start();
        Ok(Self { rules: Arc::new(rules), stacks, partial: Vec::new(), vocab, eog, trigger: None, lead_ws: false })
    }

    /// Leaves generation free until `trigger` has been generated.
    pub fn after(mut self, trigger: &str) -> Self {
        self.trigger = Some((trigger.as_bytes().to_vec(), Vec::new()));
        self
    }

    /// Advances the match by `bytes`, skipping leading whitespace right after
    /// the trigger. False if the grammar rejects them (the state is kept).
    fn consume(&mut self, bytes: &[u8]) -> bool {
        let mut bytes = bytes;
        if self.lead_ws {
            let n = bytes.iter().take_while(|b| b.is_ascii_whitespace()).count();
            bytes = &bytes[n..];
            if bytes.is_empty() {
                return true;
            }
            self.lead_ws = false;
        }
        match self.try_bytes(bytes) {
            Some((stacks, rest)) => {
                self.stacks = stacks;
                self.partial = rest;
                true
            }
            None => false,
        }
    }

    /// The stacks after `bytes` follow what was generated, or None if the
    /// grammar rejects them.
    fn try_bytes(&self, bytes: &[u8]) -> Option<(Vec<Stack>, Vec<u8>)> {
        let mut all = self.partial.clone();
        all.extend_from_slice(bytes);
        let (chars, rest) = decode(&all)?;
        let mut stacks = self.stacks.clone();
        for c in chars {
            stacks = self.rules.advance(&stacks, c);
            if stacks.is_empty() {
                return None;
            }
        }
        if !rest.is_empty() && !self.rules.takes_non_ascii(&stacks) {
            return None;
        }
        Some((stacks, rest))
    }

    /// Generation may stop here.
    pub fn can_end(&self) -> bool {
        self.trigger.is_some() || (self.partial.is_empty() && self.stacks.iter().any(|s| s.is_empty()))
    }

    fn allows(&self, token: u32) -> bool {
        if self.eog.contains(&token) {
            return self.can_end();
        }
        let Some(mut b) = self.vocab.get(token as usize).map(Vec::as_slice).filter(|b| !b.is_empty()) else {
            return false;
        };
        if self.lead_ws {
            b = &b[b.iter().take_while(|c| c.is_ascii_whitespace()).count()..];
            if b.is_empty() {
                return true;
            }
        }
        self.try_bytes(b).is_some_and(|(s, _)| !s.is_empty())
    }

    /// Sets every token the grammar does not allow next to -inf.
    pub fn mask(&self, logits: &mut [f32]) {
        const FIRST: usize = 256;
        if self.trigger.is_some() {
            return;
        }
        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        let k = FIRST.min(order.len());
        order.select_nth_unstable_by(k - 1, |&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
        let top: Vec<u32> = order[..k].iter().copied().filter(|&t| logits[t as usize].is_finite() && self.allows(t))
            .collect();
        if !top.is_empty() {
            let keep: HashSet<u32> = top.into_iter().collect();
            for (i, l) in logits.iter_mut().enumerate() {
                if !keep.contains(&(i as u32)) {
                    *l = f32::NEG_INFINITY;
                }
            }
            return;
        }
        // None of the likeliest fit: check them all.
        for (i, l) in logits.iter_mut().enumerate() {
            if l.is_finite() && !self.allows(i as u32) {
                *l = f32::NEG_INFINITY;
            }
        }
        if logits.iter().all(|l| !l.is_finite()) {
            // Nothing fits at all: let generation end rather than emit garbage.
            for &e in self.eog.iter() {
                if let Some(l) = logits.get_mut(e as usize) {
                    *l = 0.0;
                }
            }
        }
    }

    /// Advances past a sampled token.
    pub fn accept(&mut self, token: u32) {
        if self.eog.contains(&token) {
            return;
        }
        let bytes = self.vocab.get(token as usize).cloned().unwrap_or_default();
        if let Some((trigger, seen)) = &mut self.trigger {
            seen.extend_from_slice(&bytes);
            let Some(at) = seen.windows(trigger.len()).position(|w| w == trigger.as_slice()) else { return };
            let rest = seen[at + trigger.len()..].to_vec();
            self.trigger = None;
            self.lead_ws = true;
            self.consume(&rest);
            return;
        }
        self.consume(&bytes);
    }
}

// --- JSON schema -> GBNF ------------------------------------------------------------

/// llama.cpp's primitive rules (json-schema-to-grammar). Values carry no
/// whitespace of their own: it goes after an opening bracket, a comma or a
/// colon, and before a closing bracket, so nothing can follow the document
/// but the end of generation.
const PRIMITIVES: &[(&str, &str)] = &[
    ("space", r#"| " " | "\n"{1,2} [ \t]{0,20}"#),
    ("boolean", r#""true" | "false""#),
    ("null", r#""null""#),
    ("integral-part", r#"[0] | [1-9] [0-9]{0,15}"#),
    ("decimal-part", r#"[0-9]{1,16}"#),
    ("integer", r#"("-"? integral-part)"#),
    ("number", r#"("-"? integral-part) ("." decimal-part)? ([eE] [-+]? integral-part)?"#),
    ("char", r#"[^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})"#),
    ("string", r#""\"" char* "\"""#),
    ("value", r#"object | array | string | number | boolean | null"#),
    ("object", r#""{" space ( string ":" space value ("," space string ":" space value)* space )? "}""#),
    ("array", r#""[" space ( value ("," space value)* space )? "]""#),
];

/// The grammar of any JSON object (`response_format: json_object`).
pub fn json_object_grammar() -> String {
    let mut g = String::from("root ::= object\n");
    for (n, r) in PRIMITIVES {
        g.push_str(&format!("{n} ::= {r}\n"));
    }
    g
}

/// A GBNF grammar for documents matching `schema`, in llama.cpp's style.
pub fn json_schema_grammar(schema: &serde_json::Value) -> Result<String, String> {
    let mut conv = SchemaConv { rules: Vec::new(), root: schema };
    let root = conv.visit(schema, "root")?;
    let mut g = format!("root ::= {root}\n");
    for (n, r) in PRIMITIVES {
        g.push_str(&format!("{n} ::= {r}\n"));
    }
    for (n, r) in conv.rules {
        g.push_str(&format!("{n} ::= {r}\n"));
    }
    Ok(g)
}

struct SchemaConv<'a> {
    rules: Vec<(String, String)>,
    root: &'a serde_json::Value,
}

fn gbnf_literal(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\x{:02X}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl SchemaConv<'_> {
    fn add(&mut self, base: &str, body: String) -> String {
        let name = format!("{}-{}", base.replace(|c: char| !c.is_ascii_alphanumeric(), "-"), self.rules.len());
        self.rules.push((name.clone(), body));
        name
    }

    fn resolve<'b>(&'b self, r: &str) -> Option<&'b serde_json::Value> {
        let path = r.strip_prefix("#/")?;
        path.split('/').try_fold(self.root, |v, k| v.get(k))
    }

    /// A rule body (expression) for `s`.
    fn visit(&mut self, s: &serde_json::Value, name: &str) -> Result<String, String> {
        use serde_json::Value;
        if let Some(r) = s.get("$ref").and_then(Value::as_str) {
            let target = self.resolve(r).ok_or_else(|| format!("unresolved $ref {r}"))?.clone();
            return self.visit(&target, name);
        }
        if let Some(c) = s.get("const") {
            return Ok(gbnf_literal(&c.to_string()));
        }
        if let Some(Value::Array(e)) = s.get("enum") {
            let alts: Vec<String> = e.iter().map(|v| gbnf_literal(&v.to_string())).collect();
            return Ok(format!("({})", alts.join(" | ")));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(Value::Array(subs)) = s.get(key) {
                let mut alts = Vec::new();
                for (i, sub) in subs.iter().enumerate() {
                    let body = self.visit(sub, &format!("{name}-{i}"))?;
                    alts.push(self.add(name, body));
                }
                return Ok(alts.join(" | "));
            }
        }
        if let Some(Value::Array(subs)) = s.get("allOf") {
            // Merge object properties of the parts.
            let mut merged = serde_json::Map::new();
            let mut props = serde_json::Map::new();
            let mut req = Vec::new();
            for sub in subs {
                let sub = match sub.get("$ref").and_then(Value::as_str) {
                    Some(r) => self.resolve(r).cloned().unwrap_or_default(),
                    None => sub.clone(),
                };
                if let Some(Value::Object(p)) = sub.get("properties") { props.extend(p.clone()); }
                if let Some(Value::Array(r)) = sub.get("required") { req.extend(r.clone()); }
            }
            merged.insert("type".into(), "object".into());
            merged.insert("properties".into(), Value::Object(props));
            merged.insert("required".into(), Value::Array(req));
            return self.visit(&Value::Object(merged), name);
        }
        let ty = s.get("type");
        if let Some(Value::Array(types)) = ty {
            let mut alts = Vec::new();
            for (i, t) in types.iter().enumerate() {
                let mut sub = s.clone();
                sub["type"] = t.clone();
                let body = self.visit(&sub, &format!("{name}-{i}"))?;
                alts.push(self.add(name, body));
            }
            return Ok(alts.join(" | "));
        }
        let ty = ty.and_then(Value::as_str).unwrap_or(if s.get("properties").is_some() { "object" } else { "" });
        Ok(match ty {
            "object" => {
                let Some(Value::Object(props)) = s.get("properties") else { return Ok("object".into()) };
                let required: Vec<&str> = s.get("required").and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                let mut kv = Vec::new();
                for (k, v) in props {
                    let body = self.visit(v, &format!("{name}-{k}"))?;
                    let val = self.add(&format!("{name}-{k}"), body);
                    let rule = self.add(&format!("{name}-{k}-kv"),
                        format!("{} space \":\" space {val}", gbnf_literal(&format!("\"{k}\""))));
                    kv.push((k.as_str(), rule, required.contains(&k.as_str())));
                }
                let req: Vec<&String> = kv.iter().filter(|x| x.2).map(|x| &x.1).collect();
                let opt: Vec<&String> = kv.iter().filter(|x| !x.2).map(|x| &x.1).collect();
                let mut body = String::from("\"{\" space ");
                if req.is_empty() {
                    // Whichever optional property comes first, without a comma.
                    let alts: Vec<String> = (0..opt.len()).map(|i| {
                        let rest: String = opt[i + 1..].iter().map(|o| format!(" ( \",\" space {o} )?")).collect();
                        format!("{}{rest}", opt[i])
                    }).collect();
                    if !alts.is_empty() {
                        body.push_str(&format!("( {} space )?", alts.join(" | ")));
                    }
                } else {
                    body.push_str(&req.iter().map(|r| r.as_str()).collect::<Vec<_>>().join(" \",\" space "));
                    for o in &opt {
                        body.push_str(&format!(" ( \",\" space {o} )?"));
                    }
                    body.push_str(" space");
                }
                body.push_str(" \"}\"");
                body
            }
            "array" => {
                let item = match s.get("items") {
                    Some(i) => {
                        let body = self.visit(i, &format!("{name}-item"))?;
                        self.add(&format!("{name}-item"), body)
                    }
                    None => "value".into(),
                };
                let min = s.get("minItems").and_then(Value::as_u64).unwrap_or(0) as usize;
                let max = s.get("maxItems").and_then(Value::as_u64).map(|m| m as usize);
                let more = match (min.saturating_sub(1), max.map(|m| m.saturating_sub(1))) {
                    (lo, Some(hi)) => format!("(\",\" space {item}){{{lo},{hi}}}"),
                    (lo, None) => format!("(\",\" space {item}){{{lo},}}"),
                };
                if min == 0 {
                    format!("\"[\" space ( {item} {more} space )? \"]\"")
                } else {
                    format!("\"[\" space {item} {more} space \"]\"")
                }
            }
            "string" => "string".into(),
            "integer" => "integer".into(),
            "number" => "number".into(),
            "boolean" => "boolean".into(),
            "null" => "null".into(),
            _ => "value".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(words: &[&str]) -> Arc<Vec<Vec<u8>>> {
        Arc::new(words.iter().map(|w| w.as_bytes().to_vec()).collect())
    }

    fn accepts(g: &Grammar, text: &str) -> bool {
        g.try_bytes(text.as_bytes()).is_some_and(|(s, rest)| rest.is_empty() && s.iter().any(|s| s.is_empty()))
    }

    #[test]
    fn literals_alternatives_and_repetition() {
        let v = vocab(&[]);
        let g = Grammar::new(r#"root ::= " yes" | " no"   # comment
"#, v.clone(), Arc::new(vec![])).unwrap();
        assert!(accepts(&g, " yes") && accepts(&g, " no") && !accepts(&g, " maybe"));
        let g = Grammar::new("root ::= [a-c]+ (\"-\" [0-9]{2,3})?\n", v, Arc::new(vec![])).unwrap();
        for ok in ["a", "abc", "b-12", "c-123"] {
            assert!(accepts(&g, ok), "{ok}");
        }
        for bad in ["", "d", "a-1", "a-1234"] {
            assert!(!accepts(&g, bad), "{bad}");
        }
    }

    #[test]
    fn masking_keeps_only_tokens_that_fit() {
        let v = vocab(&["{", "\"a\"", ":", "1", "}", "hello", " ", "<eos>"]);
        let mut g = Grammar::new(&json_object_grammar(), v, Arc::new(vec![7])).unwrap();
        let mut logits = vec![0.0f32; 8];
        logits[5] = 5.0; // "hello" is likeliest but not JSON
        g.mask(&mut logits);
        assert!(logits[0].is_finite() && !logits[5].is_finite() && !logits[7].is_finite());
        for t in [0u32, 1, 2, 3, 4] {
            g.accept(t);
        }
        assert!(g.can_end());
    }

    #[test]
    fn schemas_constrain_keys_and_types() {
        let schema = serde_json::json!({"type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tag": {"enum": ["a", "b"]}},
            "required": ["name", "age"]});
        let g = Grammar::new(&json_schema_grammar(&schema).unwrap(), vocab(&[]), Arc::new(vec![])).unwrap();
        assert!(accepts(&g, r#"{"name": "Ana", "age": 30}"#));
        assert!(accepts(&g, "{\n  \"name\": \"Ana\",\n  \"age\": 30,\n  \"tag\": \"b\"\n}"));
        assert!(!accepts(&g, r#"{"name": "Ana"}"#));
        assert!(!accepts(&g, r#"{"name": "Ana", "age": "30"}"#));
        assert!(!accepts(&g, r#"{"name": "Ana", "age": 30, "tag": "c"}"#));
        // Nothing may follow the document.
        assert!(!accepts(&g, "{\"name\": \"Ana\", \"age\": 30}\n"));

        let optional = serde_json::json!({"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}}});
        let g = Grammar::new(&json_schema_grammar(&optional).unwrap(), vocab(&[]), Arc::new(vec![])).unwrap();
        for ok in ["{}", r#"{"a": 1}"#, r#"{"b": true}"#, r#"{"a": 1, "b": false}"#] {
            assert!(accepts(&g, ok), "{ok}");
        }
        for bad in [r#"{, "b": true}"#, r#"{"b": true, "a": 1}"#] {
            assert!(!accepts(&g, bad), "{bad}");
        }
    }

    #[test]
    fn a_triggered_grammar_waits_for_the_end_of_reasoning() {
        let v = vocab(&["hmm", "</think>", "

", "{", "}", "x"]);
        let mut g = Grammar::new(&json_object_grammar(), v, Arc::new(vec![])).unwrap().after("</think>");
        let mut logits = vec![0.0f32; 6];
        g.mask(&mut logits);
        assert!(logits.iter().all(|l| l.is_finite()), "free while thinking");
        for t in [0, 1, 2] {
            g.accept(t);
        }
        let mut logits = vec![0.0f32; 6];
        g.mask(&mut logits);
        assert!(logits[2].is_finite() && logits[3].is_finite() && !logits[5].is_finite());
        g.accept(3);
        g.accept(4);
        assert!(g.can_end());
    }

    #[test]
    fn a_character_split_across_tokens_is_carried() {
        let v = Arc::new(vec![vec![0xC3], vec![0xA9]]); // "é" in two tokens
        let mut g = Grammar::new("root ::= [^a]\n", v, Arc::new(vec![])).unwrap();
        assert!(g.allows(0));
        g.accept(0);
        assert!(!g.can_end() && g.allows(1));
        g.accept(1);
        assert!(g.can_end());
    }
}
