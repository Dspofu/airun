//! Reasoning levels: OpenAI's `reasoning_effort` (and `--reasoning-effort`)
//! for every thinking model, not only the templates that read it. Qwen3.5 /
//! 3.8 templates know only on and off, so a level becomes a token budget for
//! the `<think>` block (the reasoning budget llama-server has): when it runs
//! out, `</think>` is forced and the model answers.

/// What a level asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// No reasoning: the template's thinking switch off.
    Off,
    /// Reasoning with at most this many tokens (None: unlimited).
    Budget(Option<usize>),
}

/// The level names, cheapest first, with their budgets.
pub const LEVELS: &[(&str, Level)] = &[
    ("none", Level::Off),
    ("low", Level::Budget(Some(1024))),
    ("medium", Level::Budget(Some(4096))),
    ("high", Level::Budget(Some(16384))),
    ("xhigh", Level::Budget(Some(32768))),
    ("max", Level::Budget(None)),
];

/// `effort` as a level; OpenAI's and Anthropic's spellings are accepted.
pub fn parse(effort: &str) -> Option<Level> {
    let e = effort.trim().to_ascii_lowercase();
    let e = match e.as_str() {
        "off" | "minimal" | "disabled" | "false" | "0" => "none",
        "very_high" | "very-high" => "xhigh",
        "unlimited" | "on" | "true" | "auto" => "max",
        other => other,
    };
    LEVELS.iter().find(|(n, _)| *n == e).map(|&(_, l)| l)
}

/// The names, for help and error messages.
pub fn names() -> String {
    LEVELS.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
}

/// Parses a `--reasoning-effort` value for clap.
pub fn parse_arg(s: &str) -> Result<String, String> {
    parse(s).map(|_| s.to_ascii_lowercase()).ok_or_else(|| format!("unknown level {s:?}; use one of {}", names()))
}

/// `--reasoning on|off|auto`, as llama-server's: whether the template's
/// thinking switch is set, and which way. None is auto (the template's own).
pub fn parse_switch(s: &str) -> Result<Option<bool>, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "enabled" => Ok(Some(true)),
        "off" | "false" | "0" | "disabled" => Ok(Some(false)),
        "auto" => Ok(None),
        other => Err(format!("unknown value {other:?} for --reasoning; use on, off or auto")),
    }
}

/// Validates a `--reasoning` value for clap.
pub fn parse_switch_arg(s: &str) -> Result<String, String> {
    parse_switch(s).map(|_| s.trim().to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_their_aliases() {
        assert_eq!(parse("LOW"), Some(Level::Budget(Some(1024))));
        assert_eq!(parse("minimal"), Some(Level::Off));
        assert_eq!(parse("xhigh"), Some(Level::Budget(Some(32768))));
        assert_eq!(parse("max"), Some(Level::Budget(None)));
        assert_eq!(parse("huge"), None);
        assert_eq!(parse_switch("auto"), Ok(None));
        assert_eq!(parse_switch("off"), Ok(Some(false)));
        assert!(parse_arg("medium").is_ok() && parse_arg("x").is_err());
    }
}
