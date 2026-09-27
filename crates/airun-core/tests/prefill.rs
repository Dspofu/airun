//! Opt-in GPU integration test: a batched prefill must land where feeding the
//! same tokens one at a time lands.
//!
//! The kernels a batch takes -- integer tensor-core matmuls (MMQ) from 64
//! tokens, flash attention from 16 -- are checked one by one in `airun
//! selftest`, but `airun perplexity` feeds a token at a time and never runs
//! them on real weights. This compares the final logits of both routes. They
//! are not bit-identical (a batch quantizes activations to 8 bits; decode stays
//! f32), so the bar is agreement on the prediction and a cosine near 1.
use airun_core::{loader, LoadOptions};

const TEXT: &str = "The capital of France is Paris, and the capital of Italy is Rome. \
Water freezes at zero degrees Celsius and boils at one hundred. The sun rises in \
the east and sets in the west. A year has twelve months, and most of them have \
thirty or thirty-one days. When it rains, the streets become wet, and people open \
their umbrellas or wait indoors until the weather clears. Children go to school in \
the morning and come home in the afternoon, where they eat, do their homework, and \
play until it is time to sleep. The doctor asked how long the pain had lasted, and \
the patient said it had started three days earlier, after a long walk in the cold.";

fn check(model_var: &str, expert_cache: Option<usize>, min_cosine: f64) -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::PathBuf::from(std::env::var(model_var)?);
    let opts = LoadOptions { ctx: 1024, batch: 512, allow_fallback: false, expert_cache, ..Default::default() };
    let (mut rt, _) = loader::load(&path, &opts, |_, _, _| {})?;
    let tokens = rt.tokenizer.encode(TEXT, true);
    assert!(tokens.len() >= 100, "the passage must be long enough to take the batched kernels");

    rt.reset();
    let batched = rt.forward(&tokens)?;
    rt.reset();
    let mut sequential = Vec::new();
    for t in &tokens {
        sequential = rt.forward(std::slice::from_ref(t))?;
    }

    let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    let dot: f64 = batched.iter().zip(&sequential).map(|(a, b)| *a as f64 * *b as f64).sum();
    let na: f64 = batched.iter().map(|a| (*a as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = sequential.iter().map(|b| (*b as f64).powi(2)).sum::<f64>().sqrt();
    let cosine = dot / (na * nb);
    let worst = batched.iter().zip(&sequential).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("{model_var}: {} tokens, cosine {cosine:.6}, max |diff| {worst:.4}, top-1 {:?} vs {:?}",
        tokens.len(), argmax(&batched), argmax(&sequential));
    assert!(batched.iter().all(|x| x.is_finite()));
    assert_eq!(argmax(&batched), argmax(&sequential), "batched prefill predicts a different token");
    assert!(cosine > min_cosine, "batched prefill logits drifted: cosine {cosine}");
    Ok(())
}

#[test]
#[ignore = "requires CUDA and AIRUN_TEST_DENSE_GGUF (e.g. Qwen3-4B-Q8_0)"]
fn dense_batched_prefill_matches_sequential() -> Result<(), Box<dyn std::error::Error>> {
    // Measured 0.99983 on Qwen3-4B Q8_0.
    check("AIRUN_TEST_DENSE_GGUF", None, 0.999)
}

#[test]
#[ignore = "requires CUDA and AIRUN_TEST_MOE_GGUF (qwen35moe); streams experts"]
fn moe_batched_prefill_matches_sequential() -> Result<(), Box<dyn std::error::Error>> {
    // Routing is discrete, so 8-bit activation noise can swap an expert and
    // the difference compounds over 40 layers. Measured on Qwen3.8-35B-A3B:
    // 0.99951 with the fp16 batch kernels, 0.99784 with MMQ + flash (same
    // prediction in both). The bar sits under the second with some margin.
    check("AIRUN_TEST_MOE_GGUF", Some(96), 0.995)
}
