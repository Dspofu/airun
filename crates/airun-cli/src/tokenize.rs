//! `airun tokenize` -- inspect how a model's tokenizer splits text.
//!
//! A tokenizer that disagrees with the one a model was trained on produces
//! fluent but unrelated output, which looks like a broken forward pass. This
//! command separates the two.

use std::path::PathBuf;

use airun_core::Tokenizer;
use airun_gguf::GgufFile;
use anyhow::{Context, Result};
use clap::Args;

#[derive(Args)]
pub struct TokenizeArgs {
    /// Path to a .gguf model file.
    pub model: PathBuf,

    /// Text to tokenize.
    #[arg(short, long)]
    pub text: String,

    /// Apply the chat template before tokenizing.
    #[arg(long)]
    pub chat: bool,
}

pub fn run(args: TokenizeArgs) -> Result<()> {
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("failed to open {}", args.model.display()))?;
    let tok = Tokenizer::from_gguf(&gguf)?;

    println!(
        "tokenizer: model {:?}, pre {:?}, vocab {}",
        tok.model_kind,
        tok.pre_kind,
        tok.vocab_size()
    );
    println!(
        "  bos {:?}  eos {:?}  pad {:?}  add_bos {}",
        tok.bos, tok.eos, tok.pad, tok.add_bos
    );
    println!("  end-of-generation ids: {:?}", tok.eog);

    let text = if args.chat {
        airun_core::apply_chat_template(None, &args.text, None)
    } else {
        args.text.clone()
    };

    let ids = tok.encode(&text, true);
    println!("\ninput ({} chars): {text:?}", text.chars().count());
    println!("{} tokens:", ids.len());
    // Shown per token, so a token holding half of a character has no text to
    // show. Printing the replacement character there would look like a decoding
    // bug; printing the raw bytes says what it actually is, and seeing two of
    // them in a row is how you confirm a character was split.
    for id in &ids {
        let piece = tok.decode_one(*id);
        if piece.contains('\u{FFFD}') {
            let bytes: Vec<String> = tok
                .decode_bytes(&[*id])
                .iter()
                .map(|b| format!("{b:02X}"))
                .collect();
            println!("  {id:>7}  <{}>  (fragmento de um caractere)", bytes.join(" "));
        } else {
            println!("  {id:>7}  {piece:?}");
        }
    }

    // A tokenizer that cannot round-trip its own input is definitely wrong;
    // one that can might still disagree with the training tokenizer, but this
    // rules out the easy failure.
    let back = tok.decode(&ids, false);
    if back == text {
        println!("\nround-trip: exact");
    } else {
        println!("\nround-trip: MISMATCH");
        println!("  in:  {text:?}");
        println!("  out: {back:?}");
    }
    Ok(())
}
