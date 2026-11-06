//! `--hide-gpu` / `AIRUN_HIDE_GPUS`: GPUs airun must never touch.
//!
//! `--gpus` only chooses which cards a model goes on; the CUDA runtime can
//! still open a context (a few hundred MB of VRAM) on any visible card, GPU 0
//! above all. A hidden card is taken out of CUDA_VISIBLE_DEVICES before the
//! runtime starts, so nothing in this process can reach it.

use std::sync::OnceLock;

use anyhow::{Result, bail};

/// One line per hidden card, for `airun gpu` (empty when none is hidden).
static HIDDEN: OnceLock<Vec<String>> = OnceLock::new();

pub fn hidden() -> &'static [String] {
    HIDDEN.get().map(Vec::as_slice).unwrap_or(&[])
}

struct Card {
    ordinal: i64,
    name: String,
    uuid: String,
}

/// Hides the GPUs named by `--hide-gpu` and `AIRUN_HIDE_GPUS` (comma
/// separated: CUDA ordinals as `airun gpu` lists them with nothing hidden,
/// UUIDs, or part of a name such as `5060`). Must run before any CUDA call.
pub fn apply(flag: &[String]) -> Result<()> {
    let env = std::env::var("AIRUN_HIDE_GPUS").unwrap_or_default();
    let wanted: Vec<String> = flag.iter().map(String::as_str).chain(env.split(','))
        .map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    if wanted.is_empty() {
        return Ok(());
    }
    if std::env::var_os("CUDA_VISIBLE_DEVICES").is_some() {
        bail!("--hide-gpu / AIRUN_HIDE_GPUS cannot be combined with CUDA_VISIBLE_DEVICES, which is set; \
               unset CUDA_VISIBLE_DEVICES, or list the GPUs to keep in it and drop the hide option");
    }
    let cards = list_cards()?;
    let mut hide = vec![false; cards.len()];
    for w in &wanted {
        let lower = w.to_lowercase();
        // A number is an ordinal when some GPU has it ("1"), otherwise part
        // of a name ("5060").
        let by_ordinal: Vec<usize> = (0..cards.len())
            .filter(|&i| w.parse::<i64>().is_ok_and(|o| o == cards[i].ordinal)).collect();
        let hits: Vec<usize> = if !by_ordinal.is_empty() { by_ordinal } else {
            (0..cards.len()).filter(|&i| cards[i].uuid.eq_ignore_ascii_case(w)
                || cards[i].name.to_lowercase().contains(&lower)).collect()
        };
        if hits.is_empty() {
            let list: Vec<String> = cards.iter().map(|c| format!("{} = {}", c.ordinal, c.name)).collect();
            bail!("--hide-gpu / AIRUN_HIDE_GPUS: '{w}' matches no GPU; use a CUDA ordinal, a UUID or part \
                   of a name. The GPUs are: {}", list.join(", "));
        }
        for i in hits {
            hide[i] = true;
        }
    }
    if hide.iter().all(|&h| h) {
        bail!("--hide-gpu / AIRUN_HIDE_GPUS hides every GPU ({}); leave at least one visible",
            cards.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", "));
    }
    let keep: Vec<&Card> = cards.iter().zip(&hide).filter(|(_, h)| !**h).map(|(c, _)| c).collect();
    let visible = keep.iter().map(|c| c.uuid.as_str()).collect::<Vec<_>>().join(",");
    // SAFETY: called first thing in main, before any thread is spawned, so no
    // other thread reads the environment concurrently.
    unsafe { std::env::set_var("CUDA_VISIBLE_DEVICES", visible) };

    let hidden: Vec<String> = cards.iter().zip(&hide).filter(|(_, h)| **h)
        .map(|(c, _)| format!("GPU {} ({}, {})", c.ordinal, c.name, c.uuid)).collect();
    let renumbered: Vec<String> = keep.iter().enumerate()
        .map(|(i, c)| format!("GPU {i} = {}", c.name)).collect();
    eprintln!("note: hidden from airun: {}; the others are numbered from 0 ({}), and --gpus / \
               --main-gpu / --tensor-split use these numbers",
        hidden.join(", "), renumbered.join(", "));
    let _ = HIDDEN.set(hidden);
    Ok(())
}

/// Every CUDA device, asked of a child `airun gpu --json` so this process
/// starts the CUDA runtime only once the hidden cards are out of sight.
fn list_cards() -> Result<Vec<Card>> {
    let exe = std::env::current_exe()?;
    let out = std::process::Command::new(exe)
        .args(["gpu", "--json"])
        .env_remove("AIRUN_HIDE_GPUS")
        .env_remove("CUDA_VISIBLE_DEVICES")
        .output()?;
    if !out.status.success() {
        bail!("--hide-gpu / AIRUN_HIDE_GPUS: could not list the GPUs (`airun gpu --json` failed: {}); \
               check that `airun gpu` works", String::from_utf8_lossy(&out.stderr).trim());
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let cards = v["devices"].as_array().map(|a| a.iter().map(|d| Card {
        ordinal: d["ordinal"].as_i64().unwrap_or(-1),
        name: d["name"].as_str().unwrap_or("").to_string(),
        uuid: d["uuid"].as_str().unwrap_or("").to_string(),
    }).collect::<Vec<_>>()).unwrap_or_default();
    if cards.is_empty() {
        bail!("--hide-gpu / AIRUN_HIDE_GPUS: no CUDA device found; check that `airun gpu` lists your GPUs");
    }
    Ok(cards)
}
