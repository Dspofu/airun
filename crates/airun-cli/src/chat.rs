//! Chat templates in, structured replies out -- what llama.cpp's server does
//! with `--jinja`, so a client sees the same messages from either.
//!
//! The prompt is the GGUF's own Jinja chat template run over the request's
//! messages and tools (minijinja with Python string methods), so tool
//! definitions, tool results, system prompts and switches such as
//! `enable_thinking` or `reasoning_effort` read exactly as the model was
//! trained on. The reply is split back into `reasoning_content` (the `<think>`
//! block), `content`, and `tool_calls` (Qwen's `<tool_call>` blocks, in the
//! XML form `<function=..><parameter=..>` of Qwen3.5+ or the JSON form of
//! Qwen2.5/3).

use anyhow::{Context, Result, anyhow};
use serde_json::{Map, Value, json};

pub struct ChatTemplate {
    env: minijinja::Environment<'static>,
    pub bos: String,
    pub eos: String,
}

impl ChatTemplate {
    pub fn new(source: String, bos: String, eos: String) -> Result<Self> {
        let mut env = minijinja::Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_function("raise_exception", |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, msg))
        });
        env.add_function("strftime_now", |fmt: String| -> String {
            // Templates use it for "today's date" lines; the server's local
            // date in the requested format would need a time crate, and the
            // common formats only need the date.
            let days = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() / 86_400).unwrap_or(0) as i64;
            let (y, m, d) = civil_from_days(days);
            const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"];
            fmt.replace("%Y", &y.to_string()).replace("%m", &format!("{m:02}")).replace("%d", &format!("{d:02}"))
                .replace("%B", MONTHS[(m - 1) as usize]).replace("%b", &MONTHS[(m - 1) as usize][..3])
        });
        env.add_template_owned("chat", source)
            .map_err(|e| anyhow!("the model's chat template does not parse: {e}"))?;
        Ok(Self { env, bos, eos })
    }

    /// The prompt for `messages` (OpenAI shape) and `tools`, with the
    /// template's switches from `kwargs` (enable_thinking, reasoning_effort,
    /// ...). Assistant tool-call arguments sent as JSON strings are parsed
    /// into objects first, as llama.cpp does, since templates iterate them.
    pub fn render(&self, messages: &Value, tools: Option<&Value>, add_generation_prompt: bool,
        kwargs: &Map<String, Value>) -> Result<String> {
        let mut ctx = kwargs.clone();
        ctx.insert("messages".into(), normalize_messages(messages));
        if let Some(t) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            ctx.insert("tools".into(), t.clone());
        }
        ctx.insert("add_generation_prompt".into(), Value::Bool(add_generation_prompt));
        ctx.insert("bos_token".into(), Value::String(self.bos.clone()));
        ctx.insert("eos_token".into(), Value::String(self.eos.clone()));
        let tmpl = self.env.get_template("chat").context("chat template")?;
        tmpl.render(minijinja::Value::from_serialize(&ctx))
            .map_err(|e| anyhow!("the chat template failed on this request: {e}"))
    }
}

fn normalize_messages(messages: &Value) -> Value {
    let Some(list) = messages.as_array() else { return messages.clone() };
    Value::Array(list.iter().map(|m| {
        let mut m = m.clone();
        if let Some(obj) = m.as_object_mut() {
            if obj.get("content").is_none_or(Value::is_null) {
                obj.insert("content".into(), Value::String(String::new()));
            }
            if let Some(Value::Array(calls)) = obj.get_mut("tool_calls") {
                for call in calls {
                    if let Some(f) = call.get_mut("function").and_then(Value::as_object_mut) {
                        if let Some(Value::String(args)) = f.get("arguments") {
                            let parsed = serde_json::from_str::<Value>(args).unwrap_or_else(|_| json!({}));
                            f.insert("arguments".into(), parsed);
                        }
                    }
                }
            }
        }
        m
    }).collect())
}

/// Days since 1970-01-01 to (year, month, day), Howard Hinnant's algorithm.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// One function call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// The arguments as a JSON object.
    pub arguments: Value,
}

impl ToolCall {
    /// OpenAI's shape; `arguments` is a JSON string there.
    pub fn to_json(&self, index: usize, id: &str) -> Value {
        json!({
            "type": "function",
            "id": id,
            "index": index,
            "function": { "name": self.name, "arguments": self.arguments.to_string() },
        })
    }
}

/// A reply split the way llama.cpp's server returns it.
#[derive(Debug, Default, PartialEq)]
pub struct Reply {
    pub reasoning: Option<String>,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Splits `text` (what the model wrote after the prompt). `opened`: the prompt
/// itself opened a `<think>` block, so the reply starts inside it.
/// `extract_reasoning` false leaves the thinking in `content` (llama.cpp's
/// `reasoning_format: none`).
pub fn parse_reply(text: &str, opened: bool, extract_reasoning: bool) -> Reply {
    let (reasoning, rest) = split_reasoning(text, opened);
    let (content, tool_calls) = split_tool_calls(rest);
    if extract_reasoning {
        Reply { reasoning, content: content.trim().to_string(), tool_calls }
    } else {
        let content = if tool_calls.is_empty() { text.to_string() } else {
            text[..text.find("<tool_call>").unwrap_or(text.len())].to_string()
        };
        Reply { reasoning: None, content: content.trim_end().to_string(), tool_calls }
    }
}

/// [`parse_reply`] for `tool_choice: "none"`: a call the model writes anyway
/// stays in the content as text, as llama.cpp leaves it.
pub fn parse_reply_without_tools(text: &str, opened: bool, extract_reasoning: bool) -> Reply {
    if !extract_reasoning {
        return Reply { reasoning: None, content: text.trim_end().to_string(), tool_calls: Vec::new() };
    }
    let (reasoning, rest) = split_reasoning(text, opened);
    Reply { reasoning, content: rest.trim().to_string(), tool_calls: Vec::new() }
}

fn split_reasoning(text: &str, opened: bool) -> (Option<String>, &str) {
    let body = if opened { Some(text) } else { text.trim_start().strip_prefix("<think>") };
    match body {
        Some(b) => match b.find("</think>") {
            Some(end) => (Some(b[..end].trim().to_string()), &b[end + "</think>".len()..]),
            // Still thinking when it stopped: all of it is reasoning.
            None => (Some(b.trim().to_string()), ""),
        },
        None => (None, text),
    }
}

/// `content` before the first tool call, and the calls.
fn split_tool_calls(text: &str) -> (&str, Vec<ToolCall>) {
    let Some(first) = text.find("<tool_call>") else { return (text, Vec::new()) };
    let mut calls = Vec::new();
    let mut rest = &text[first..];
    while let Some(start) = rest.find("<tool_call>") {
        let body_start = start + "<tool_call>".len();
        let (body, next) = match rest[body_start..].find("</tool_call>") {
            Some(end) => (&rest[body_start..body_start + end], &rest[body_start + end + "</tool_call>".len()..]),
            None => (&rest[body_start..], ""),
        };
        if let Some(c) = parse_tool_call(body) {
            calls.push(c);
        }
        rest = next;
    }
    (&text[..first], calls)
}

fn parse_tool_call(body: &str) -> Option<ToolCall> {
    let body = body.trim();
    if let Some(f) = body.strip_prefix("<function=") {
        // <function=NAME>\n<parameter=P>\nVALUE\n</parameter>...</function>
        let name_end = f.find('>')?;
        let name = f[..name_end].trim().to_string();
        let mut args = Map::new();
        let mut rest = &f[name_end + 1..];
        while let Some(p) = rest.find("<parameter=") {
            let after = &rest[p + "<parameter=".len()..];
            let key_end = after.find('>')?;
            let key = after[..key_end].trim().to_string();
            let val_start = key_end + 1;
            let val_end = after[val_start..].find("</parameter>").map_or(after.len(), |e| val_start + e);
            let raw = after[val_start..val_end].trim_matches('\n');
            // A value that reads as JSON (a number, object, ...) is that value.
            let value = match serde_json::from_str::<Value>(raw.trim()) {
                Ok(v) if !v.is_string() => v,
                _ => Value::String(raw.to_string()),
            };
            args.insert(key, value);
            rest = &after[(val_end + "</parameter>".len()).min(after.len())..];
        }
        return Some(ToolCall { name, arguments: Value::Object(args) });
    }
    // {"name": ..., "arguments": {...}}
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = match v.get("arguments").or_else(|| v.get("parameters")) {
        Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
        Some(a) => a.clone(),
        None => json!({}),
    };
    Some(ToolCall { name, arguments })
}

/// Turns the growing reply into stream deltas: reasoning and content as they
/// arrive, text from a `<tool_call>` on held back for [`Self::finish`].
pub struct DeltaSplitter {
    extract_reasoning: bool,
    in_reasoning: bool,
    /// Bytes of the reply already emitted (or consumed as markup).
    done: usize,
    /// A tool call has started: no more content goes out.
    in_tools: bool,
    /// Tool calls are recognised at all (not with `tool_choice: "none"`).
    tools: bool,
    /// Just past `</think>`: the blank line before the answer, which may come
    /// in later tokens, is dropped (llama.cpp's content starts at the answer).
    skip_ws: bool,
}

/// One piece of a streamed reply.
#[derive(Debug, PartialEq)]
pub enum Delta {
    Reasoning(String),
    Content(String),
}

const THINK_END: &str = "</think>";
const TOOL_START: &str = "<tool_call>";

impl DeltaSplitter {
    pub fn new(opened: bool, extract_reasoning: bool) -> Self {
        Self { extract_reasoning, in_reasoning: opened && extract_reasoning, done: 0, in_tools: false,
            tools: true, skip_ws: false }
    }

    /// Streams `<tool_call>` text as content instead of holding it back.
    pub fn without_tools(mut self) -> Self {
        self.tools = false;
        self
    }

    /// Deltas for `text`, the whole reply so far; `hold` trailing bytes may
    /// still turn into a stop sequence and are not released yet.
    pub fn feed(&mut self, text: &str, hold: usize) -> Vec<Delta> {
        let mut out = Vec::new();
        let limit = text.len().saturating_sub(hold);
        loop {
            if self.in_tools || self.done >= limit {
                return out;
            }
            let pending = &text[self.done..limit];
            if self.skip_ws {
                let t = pending.trim_start();
                self.done += pending.len() - t.len();
                if t.is_empty() {
                    return out;
                }
                self.skip_ws = false;
                continue;
            }
            if self.extract_reasoning && !self.in_reasoning && self.done == 0 {
                // A reply that opens its own <think> block.
                let t = pending.trim_start();
                if t.starts_with("<think>") {
                    self.done += pending.len() - t.len() + "<think>".len();
                    self.in_reasoning = true;
                    continue;
                }
                if "<think>".starts_with(t) && !t.is_empty() {
                    return out; // could still become <think>
                }
            }
            if !self.in_reasoning && !self.tools {
                let cut = floor_char(pending, pending.len());
                push(&mut out, false, &pending[..cut]);
                self.done += cut;
                return out;
            }
            let marker = if self.in_reasoning { THINK_END } else { TOOL_START };
            if let Some(at) = pending.find(marker) {
                push(&mut out, self.in_reasoning, &pending[..at]);
                self.done += at + marker.len();
                if self.in_reasoning {
                    self.in_reasoning = false;
                    self.skip_ws = true;
                } else {
                    self.in_tools = true;
                }
                continue;
            }
            // Hold back a tail that could be the start of the marker.
            let keep = (1..marker.len()).rev().find(|&n| pending.ends_with(&marker[..n])).unwrap_or(0);
            let cut = floor_char(pending, pending.len() - keep);
            push(&mut out, self.in_reasoning, &pending[..cut]);
            self.done += cut;
            return out;
        }
    }
}

fn push(out: &mut Vec<Delta>, reasoning: bool, s: &str) {
    if !s.is_empty() {
        out.push(if reasoning { Delta::Reasoning(s.to_string()) } else { Delta::Content(s.to_string()) });
    }
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_xml_tool_calls_parse() {
        let text = "Let me check.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
        let r = parse_reply(text, false, true);
        assert_eq!(r.content, "Let me check.");
        assert_eq!(r.tool_calls, vec![ToolCall { name: "get_weather".into(),
            arguments: json!({"city": "Paris", "days": 3}) }]);
    }

    #[test]
    fn json_tool_calls_and_reasoning_parse() {
        let text = "hmm, weather\n</think>\n\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>";
        let r = parse_reply(text, true, true);
        assert_eq!(r.reasoning.as_deref(), Some("hmm, weather"));
        assert_eq!(r.content, "");
        assert_eq!(r.tool_calls[0].name, "f");
        let plain = parse_reply("<think>\nx\n</think>\n\nParis.", false, true);
        assert_eq!((plain.reasoning.as_deref(), plain.content.as_str()), (Some("x"), "Paris."));
        assert_eq!(parse_reply("Paris.", false, true).reasoning, None);
    }

    #[test]
    fn deltas_split_reasoning_content_and_hold_tool_calls() {
        let full = "think a bit</think>\n\nAnswer here<tool_call>\n<function=f>\n</function>\n</tool_call>";
        let mut s = DeltaSplitter::new(true, true);
        let mut got = Vec::new();
        for end in 1..=full.len() {
            if full.is_char_boundary(end) {
                got.extend(s.feed(&full[..end], 0));
            }
        }
        let reasoning: String = got.iter().filter_map(|d| match d { Delta::Reasoning(r) => Some(r.as_str()), _ => None }).collect();
        let content: String = got.iter().filter_map(|d| match d { Delta::Content(c) => Some(c.as_str()), _ => None }).collect();
        assert_eq!(reasoning, "think a bit");
        assert_eq!(content, "Answer here");
    }

    #[test]
    fn the_qwen38_template_renders_tools_and_results() {
        let src = "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}{%- if m.tool_calls %}\
                   {%- for c in m.tool_calls %}<tool_call>{{ c.function.name }}{{ c.function.arguments | tojson }}\
                   {%- endfor %}{%- endif %}<|im_end|>\n{%- endfor %}{%- if tools %}T{{ tools | length }}{%- endif %}\
                   {%- if add_generation_prompt %}<|im_start|>assistant\n{%- endif %}{{ 'x'.startswith('x') }}";
        let t = ChatTemplate::new(src.into(), String::new(), String::new()).unwrap();
        let msgs = json!([{"role": "user", "content": "hi"},
            {"role": "assistant", "content": null, "tool_calls": [{"type": "function",
                "function": {"name": "f", "arguments": "{\"a\":1}"}}]}]);
        let out = t.render(&msgs, Some(&json!([{"type": "function"}])), true, &Map::new()).unwrap();
        assert!(out.contains("<tool_call>f{\"a\":1}"), "{out}");
        assert!(out.contains("T1") && out.ends_with("<|im_start|>assistantTrue"), "{out}");
    }

    #[test]
    fn dates_convert() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_757), (2026, 10, 31));
    }
}
