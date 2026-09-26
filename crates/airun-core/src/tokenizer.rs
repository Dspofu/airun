//! Byte-level BPE tokenizer, built from the vocabulary embedded in a GGUF file.
//!
//! This is the GPT-2 scheme that Qwen, Llama-3 and most recent models use: bytes
//! are first mapped into a printable Unicode alphabet, the text is split by a
//! pre-tokenizer pattern, and merges are applied within each piece. Getting the
//! pre-tokenizer wrong produces subtly different token boundaries and therefore
//! subtly wrong output, so the split rules are implemented explicitly rather
//! than approximated.

use std::collections::HashMap;

use airun_gguf::GgufFile;

use crate::error::{CoreError, Result};

/// GGUF token type tags.
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_USER_DEFINED: i64 = 4;
const TOKEN_TYPE_BYTE: i64 = 6;

/// The GPT-2 byte alphabet: a bijection from all 256 byte values onto printable
/// code points, so byte sequences survive as text through the merge process.
fn byte_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut assigned = [false; 256];

    // These ranges map to themselves.
    for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
        table[b as usize] = char::from_u32(b as u32).unwrap();
        assigned[b as usize] = true;
    }
    // Everything else is pushed above the BMP control area, in order.
    let mut n = 0u32;
    for b in 0..256usize {
        if !assigned[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

pub struct Tokenizer {
    /// id -> token text, in the byte-encoded alphabet.
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    /// (left, right) -> rank; lower merges first.
    merges: HashMap<(String, String), u32>,
    /// Literal strings that must be matched before ordinary pre-tokenization.
    special: Vec<(String, u32)>,

    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,

    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pad: Option<u32>,
    /// Additional ids that should stop generation, e.g. `<|im_end|>`.
    pub eog: Vec<u32>,
    pub add_bos: bool,

    pub model_kind: String,
    pub pre_kind: String,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let meta = gguf.metadata();

        let model_kind = meta
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("gpt2")
            .to_string();
        if model_kind != "gpt2" && model_kind != "llama" {
            return Err(CoreError::Tokenizer(format!(
                "tokenizer model {model_kind:?} is not supported; airun implements byte-level \
                 BPE (`gpt2`)"
            )));
        }

        let tokens: Vec<String> = meta
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_strings())
            .map(|s| s.to_vec())
            .ok_or_else(|| {
                CoreError::Tokenizer("model has no tokenizer.ggml.tokens array".into())
            })?;

        let token_types: Vec<i64> = meta
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_i64_vec())
            .unwrap_or_else(|| vec![1; tokens.len()]);

        let merge_list: Vec<String> = meta
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .and_then(|a| a.as_strings())
            .map(|s| s.to_vec())
            .unwrap_or_default();

        let mut merges = HashMap::with_capacity(merge_list.len());
        for (rank, m) in merge_list.iter().enumerate() {
            // Entries are "left right"; a token containing a space would be
            // ambiguous, but the byte alphabet never emits a literal space.
            if let Some((l, r)) = m.split_once(' ') {
                merges.insert((l.to_string(), r.to_string()), rank as u32);
            }
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, t) in tokens.iter().enumerate() {
            // First occurrence wins, matching how upstream builds its map.
            token_to_id.entry(t.clone()).or_insert(id as u32);
        }

        // Control and user-defined tokens are matched literally in the input.
        // Longest first, so `<|im_start|>` wins over a shorter prefix.
        let mut special: Vec<(String, u32)> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    token_types.get(*i).copied().unwrap_or(1),
                    TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
                )
            })
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        special.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let byte_encoder = byte_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let get_id = |key: &str| meta.get(key).and_then(|v| v.as_u64()).map(|v| v as u32);
        let eos = get_id("tokenizer.ggml.eos_token_id");

        // Any control token that looks like an end-of-turn marker also stops
        // generation; chat models emit these instead of the raw EOS.
        let mut eog: Vec<u32> = special
            .iter()
            .filter(|(t, _)| {
                matches!(
                    t.as_str(),
                    "<|im_end|>" | "<|endoftext|>" | "<|eot_id|>" | "<end_of_turn>"
                )
            })
            .map(|(_, id)| *id)
            .collect();
        if let Some(e) = eos {
            if !eog.contains(&e) {
                eog.push(e);
            }
        }

        // Byte-fallback tokens must exist for the encoder to be total.
        let byte_tokens = token_types
            .iter()
            .filter(|&&t| t == TOKEN_TYPE_BYTE)
            .count();

        let tok = Self {
            tokens,
            token_to_id,
            merges,
            special,
            byte_encoder,
            byte_decoder,
            bos: get_id("tokenizer.ggml.bos_token_id"),
            eos,
            pad: get_id("tokenizer.ggml.padding_token_id"),
            eog,
            add_bos: meta
                .get("tokenizer.ggml.add_bos_token")
                .and_then(|v| v.as_bool())
                .unwrap_or(false),
            model_kind,
            pre_kind: meta
                .get("tokenizer.ggml.pre")
                .and_then(|v| v.as_str())
                .unwrap_or("default")
                .to_string(),
        };

        if tok.merges.is_empty() && byte_tokens == 0 {
            return Err(CoreError::Tokenizer(
                "model has neither BPE merges nor byte-fallback tokens; it cannot be encoded"
                    .into(),
            ));
        }
        Ok(tok)
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// Encodes text into token ids.
    ///
    /// Special tokens present in the text are matched literally, which is what
    /// makes a pre-formatted chat prompt tokenize the same way the model was
    /// trained on.
    pub fn encode(&self, text: &str, add_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_special && self.add_bos {
            if let Some(b) = self.bos {
                out.push(b);
            }
        }
        self.encode_into(text, &mut out);
        out
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Carve out special tokens, then BPE the ordinary text between them.
        if let Some((start, len, id)) = self.find_special(text) {
            self.encode_ordinary(&text[..start], out);
            out.push(id);
            self.encode_into(&text[start + len..], out);
            return;
        }
        self.encode_ordinary(text, out);
    }

    fn find_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (tok, id) in &self.special {
            if let Some(pos) = text.find(tok.as_str()) {
                // Earliest match wins; `special` is length-sorted so the longest
                // alternative at the same position is found first.
                let better = match best {
                    None => true,
                    Some((bp, bl, _)) => pos < bp || (pos == bp && tok.len() > bl),
                };
                if better {
                    best = Some((pos, tok.len(), *id));
                }
            }
        }
        best
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        for piece in pretokenize(text) {
            // Byte-level encode, then merge.
            let encoded: String = piece
                .bytes()
                .map(|b| self.byte_encoder[b as usize])
                .collect();
            self.bpe_into(&encoded, out);
        }
    }

    /// Applies merges to one pre-token and appends the resulting ids.
    fn bpe_into(&self, word: &str, out: &mut Vec<u32>) {
        if word.is_empty() {
            return;
        }
        // A whole-word hit is common enough to be worth checking first.
        if let Some(&id) = self.token_to_id.get(word) {
            out.push(id);
            return;
        }

        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            // Find the adjacent pair with the lowest merge rank.
            let mut best: Option<(usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merges.get(&key) {
                    if best.is_none_or(|(_, r)| rank < r) {
                        best = Some((i, rank));
                    }
                }
            }
            let Some((i, _)) = best else { break };

            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..i + 2, [merged]);
        }

        for sym in symbols {
            if let Some(&id) = self.token_to_id.get(&sym) {
                out.push(id);
            } else {
                // No token for this symbol: fall back to per-byte tokens so
                // encoding is always total, never silently lossy.
                self.push_byte_fallback(&sym, out);
            }
        }
    }

    fn push_byte_fallback(&self, sym: &str, out: &mut Vec<u32>) {
        for c in sym.chars() {
            if let Some(&b) = self.byte_decoder.get(&c) {
                // `<0xNN>` is the conventional spelling of a byte token.
                if let Some(&id) = self.token_to_id.get(&format!("<0x{b:02X}>")) {
                    out.push(id);
                    continue;
                }
            }
            if let Some(&id) = self.token_to_id.get(&c.to_string()) {
                out.push(id);
            }
        }
    }

    /// Decodes ids back into text.
    ///
    /// Returns bytes reassembled through the inverse byte alphabet, so multi-byte
    /// UTF-8 split across tokens is rejoined correctly.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.token_bytes(id, skip_special, &mut bytes);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Appends one token's raw bytes to `out`.
    ///
    /// Bytes, not text: a single token is often half of a character, so there is
    /// nothing to turn into a `String` yet. Only the caller, once it has enough
    /// tokens, knows where a character ends.
    fn token_bytes(&self, id: u32, skip_special: bool, out: &mut Vec<u8>) {
        let Some(text) = self.token_text(id) else {
            return;
        };
        if skip_special && self.special.iter().any(|(_, sid)| *sid == id) {
            return;
        }
        if let Some(byte) = parse_byte_token(text) {
            out.push(byte);
            return;
        }
        for c in text.chars() {
            match self.byte_decoder.get(&c) {
                Some(&b) => out.push(b),
                // Not in the byte alphabet: emit the character as-is.
                None => {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
        }
    }

    /// The raw bytes `ids` decode to, before any UTF-8 interpretation.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.token_bytes(id, false, &mut out);
        }
        out
    }

    /// Decodes a single token in isolation.
    ///
    /// Only correct for tokens that are whole characters. Streaming output must
    /// use [`StreamDecoder`], which carries the partial ones across tokens.
    pub fn decode_one(&self, id: u32) -> String {
        self.decode(&[id], false)
    }

    /// An incremental decoder for streaming generation.
    pub fn stream_decoder(&self) -> StreamDecoder {
        StreamDecoder::default()
    }
}

/// Reassembles text from tokens arriving one at a time.
///
/// A byte-level BPE puts character boundaries wherever the merge table happens
/// to, so a single accented letter routinely spans two tokens -- `"Órbita"`
/// tokenizes with `"Ó"` split down the middle. Decoding each token on its own
/// runs `from_utf8_lossy` over half a character, which yields `U+FFFD` and
/// destroys the bytes, so the character can never be recovered even once its
/// other half arrives. Any non-ASCII output is corrupted that way: accented
/// Latin, CJK, emoji.
///
/// This holds the trailing bytes of an incomplete sequence until the token that
/// completes them shows up.
#[derive(Default)]
pub struct StreamDecoder {
    /// Bytes seen but not yet forming a complete character.
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// Feeds one token and returns whatever text is now complete, which is the
    /// empty string when the token was only part of a character.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> String {
        tokenizer.token_bytes(id, false, &mut self.pending);
        self.take_complete()
    }

    /// Emits any bytes still held, as replacement characters if they never
    /// completed. Call once at the end of a generation.
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }

    fn take_complete(&mut self) -> String {
        match std::str::from_utf8(&self.pending) {
            Ok(_) => {
                // SAFETY-free path: validated just above, so this cannot fail.
                let out = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                out
            }
            Err(e) => {
                let valid = e.valid_up_to();
                // `error_len` is None when the tail is merely unfinished; that
                // is the case worth waiting on. Some(n) means those bytes can
                // never become valid, so they are consumed as replacements now
                // rather than blocking every token after them.
                let consume = match e.error_len() {
                    Some(n) => valid + n,
                    None => valid,
                };
                let out = String::from_utf8_lossy(&self.pending[..consume]).into_owned();
                self.pending.drain(..consume);
                out
            }
        }
    }
}

/// Recognises `<0xNN>` byte tokens.
fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

/// Splits text the way the GPT-2 / Qwen pre-tokenizer regex does.
///
/// The reference pattern is:
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
/// It is implemented directly rather than with a regex engine: the alternation
/// order is what defines token boundaries, and an approximation would shift them.
fn pretokenize(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pieces = Vec::new();
    let mut i = 0usize;

    let char_at = |k: usize| chars.get(k).map(|(_, c)| *c);
    let byte_at = |k: usize| chars.get(k).map(|(b, _)| *b).unwrap_or(bytes.len());

    while i < chars.len() {
        let start = i;

        // 1. Contractions: 's 't 're 've 'm 'll 'd, case-insensitive.
        if char_at(i) == Some('\'') {
            if let Some(len) = match_contraction(&chars, i) {
                i += len;
                pieces.push(&text[byte_at(start)..byte_at(i)]);
                continue;
            }
        }

        // 2. An optional non-letter/digit lead-in followed by letters.
        {
            let mut j = i;
            let c = char_at(j).unwrap();
            let lead = !c.is_alphanumeric() && c != '\r' && c != '\n' && !is_digit(c);
            if lead && char_at(j + 1).is_some_and(is_letter) {
                j += 1;
            }
            if char_at(j).is_some_and(is_letter) {
                while char_at(j).is_some_and(is_letter) {
                    j += 1;
                }
                i = j;
                pieces.push(&text[byte_at(start)..byte_at(i)]);
                continue;
            }
        }

        // 3. A single digit. Splitting digits one at a time is deliberate: it is
        //    what the pattern does, and it is why models handle arithmetic the
        //    way they do.
        if char_at(i).is_some_and(is_digit) {
            i += 1;
            pieces.push(&text[byte_at(start)..byte_at(i)]);
            continue;
        }

        // 4. An optional space, then punctuation, then any trailing newlines.
        {
            let mut j = i;
            if char_at(j) == Some(' ') && char_at(j + 1).is_some_and(is_punct) {
                j += 1;
            }
            if char_at(j).is_some_and(is_punct) {
                while char_at(j).is_some_and(is_punct) {
                    j += 1;
                }
                while matches!(char_at(j), Some('\r') | Some('\n')) {
                    j += 1;
                }
                i = j;
                pieces.push(&text[byte_at(start)..byte_at(i)]);
                continue;
            }
        }

        // 5. Whitespace runs ending in a newline.
        {
            let mut j = i;
            while char_at(j).is_some_and(|c| c.is_whitespace() && c != '\r' && c != '\n') {
                j += 1;
            }
            if matches!(char_at(j), Some('\r') | Some('\n')) {
                while matches!(char_at(j), Some('\r') | Some('\n')) {
                    j += 1;
                }
                i = j;
                pieces.push(&text[byte_at(start)..byte_at(i)]);
                continue;
            }
        }

        // 6. Whitespace not followed by a non-space, then plain whitespace.
        //    The lookahead keeps the space that belongs to the next word out of
        //    this piece -- " hello" must tokenize as one unit, not " " + "hello".
        if char_at(i).is_some_and(char::is_whitespace) {
            let mut j = i;
            while char_at(j).is_some_and(char::is_whitespace) {
                j += 1;
            }
            // Give the final space back when a non-space follows.
            if j > i && char_at(j).is_some() && j - 1 > i {
                j -= 1;
            }
            if j == i {
                j = i + 1;
            }
            i = j;
            pieces.push(&text[byte_at(start)..byte_at(i)]);
            continue;
        }

        // Anything unmatched advances by one so the loop always terminates.
        i += 1;
        pieces.push(&text[byte_at(start)..byte_at(i)]);
    }

    pieces
}

fn match_contraction(chars: &[(usize, char)], i: usize) -> Option<usize> {
    let next = |k: usize| chars.get(k).map(|(_, c)| c.to_ascii_lowercase());
    match (next(i + 1), next(i + 2)) {
        (Some('s'), _) | (Some('t'), _) | (Some('m'), _) | (Some('d'), _) => Some(2),
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => Some(3),
        _ => None,
    }
}

fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

fn is_digit(c: char) -> bool {
    c.is_numeric()
}

fn is_punct(c: char) -> bool {
    !c.is_whitespace() && !c.is_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_alphabet_is_a_bijection() {
        let enc = byte_to_unicode();
        let mut seen = std::collections::HashSet::new();
        for c in enc {
            assert!(seen.insert(c), "byte alphabet repeated {c:?}");
        }
        assert_eq!(seen.len(), 256);
        // Printable ASCII maps to itself.
        assert_eq!(enc[b'A' as usize], 'A');
        assert_eq!(enc[b'~' as usize], '~');
        // Space does not, or it would be ambiguous inside merge entries.
        assert_ne!(enc[b' ' as usize], ' ');
    }

    #[test]
    fn pretokenizer_keeps_leading_spaces_with_words() {
        assert_eq!(pretokenize("hello world"), vec!["hello", " world"]);
        assert_eq!(pretokenize("a b c"), vec!["a", " b", " c"]);
    }

    #[test]
    fn pretokenizer_splits_digits_individually() {
        assert_eq!(pretokenize("123"), vec!["1", "2", "3"]);
        assert_eq!(pretokenize("x42"), vec!["x", "4", "2"]);
    }

    #[test]
    fn pretokenizer_handles_contractions() {
        assert_eq!(pretokenize("don't"), vec!["don", "'t"]);
        assert_eq!(pretokenize("we're"), vec!["we", "'re"]);
        assert_eq!(pretokenize("I'll"), vec!["I", "'ll"]);
    }

    #[test]
    fn pretokenizer_groups_punctuation_and_newlines() {
        assert_eq!(pretokenize("hi!!!"), vec!["hi", "!!!"]);
        assert_eq!(pretokenize("a.\n"), vec!["a", ".\n"]);
    }

    #[test]
    fn pretokenizer_covers_every_byte_of_the_input() {
        // Losing or duplicating input would corrupt the prompt silently.
        for text in [
            "hello world",
            "  leading",
            "trailing  ",
            "mixed 123 !@# \n\n text",
            "unicode: café 日本語",
            "",
        ] {
            let joined: String = pretokenize(text).concat();
            assert_eq!(joined, text, "pretokenizer lost data on {text:?}");
        }
    }

    #[test]
    fn byte_token_parsing() {
        assert_eq!(parse_byte_token("<0x0A>"), Some(10));
        assert_eq!(parse_byte_token("<0xFF>"), Some(255));
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xZZ>"), None);
    }

    /// Feeds bytes to the stream decoder one "token" at a time.
    fn stream(chunks: &[&[u8]]) -> String {
        let mut d = StreamDecoder::default();
        let mut out = String::new();
        for chunk in chunks {
            d.pending.extend_from_slice(chunk);
            out.push_str(&d.take_complete());
        }
        out.push_str(&d.flush());
        out
    }

    #[test]
    fn streaming_rejoins_a_character_split_across_tokens() {
        // How a byte-level BPE actually emits "Ó": one token carries the lead
        // byte, the next carries the continuation. Decoding either alone would
        // produce a replacement character and lose the letter for good.
        assert_eq!(stream(&[&[0xC3], &[0x93]]), "Ó");
        assert_eq!(stream(&[b" ", &[0xC3], &[0x93], b"rbita"]), " Órbita");

        // Three- and four-byte sequences, split at every position.
        assert_eq!(stream(&[&[0xE2], &[0x82], &[0xAC]]), "€");
        assert_eq!(stream(&[&[0xE2, 0x82], &[0xAC]]), "€");
        assert_eq!(stream(&[&[0xF0], &[0x9F, 0x92], &[0xA1]]), "💡");
    }

    #[test]
    fn streaming_matches_decoding_the_whole_sequence() {
        let text = "Café Órbita — 你好 💡 coração";
        let bytes = text.as_bytes();
        // Split at every byte boundary; the result must not depend on where.
        for cut in 1..bytes.len() {
            let joined = stream(&[&bytes[..cut], &bytes[cut..]]);
            assert_eq!(joined, text, "streaming diverged when split at {cut}");
        }
    }

    #[test]
    fn streaming_does_not_stall_on_bytes_that_can_never_be_valid() {
        // A lone continuation byte is not the start of anything. Holding it
        // would swallow every token after it, so it is emitted as a
        // replacement and the stream keeps going.
        assert_eq!(stream(&[&[0x80], b"ok"]), "\u{FFFD}ok");
        // Trailing incomplete bytes surface at flush rather than vanishing.
        assert_eq!(stream(&[b"ab", &[0xC3]]), "ab\u{FFFD}");
    }
}
