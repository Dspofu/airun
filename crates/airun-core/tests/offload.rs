//! Opt-in GPU integration test, using four real hybrid/MoE layers. This checks
//! composition, cache eviction, prefill and reset against resident weights.
use airun_core::{loader, LoadOptions};
use airun_gguf::{GgufFile, GgufWriter};

#[test]
#[ignore = "requires CUDA and AIRUN_TEST_MOE_GGUF; writes a temporary four-layer fixture"]
fn cached_experts_match_resident_logits() -> Result<(), Box<dyn std::error::Error>> {
    let source = GgufFile::open(std::env::var("AIRUN_TEST_MOE_GGUF")?)?;
    let path = std::env::current_dir()?.join(format!("offload-test-{}.gguf", std::process::id()));
    struct Cleanup(std::path::PathBuf);
    impl Drop for Cleanup { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
    let _cleanup = Cleanup(path.clone());
    let mut writer = GgufWriter::create(&path)?;
    for (key, value) in source.metadata() { writer.set_metadata(key, value.clone()); }
    let arch = source.architecture().unwrap();
    writer.set_u32(format!("{arch}.block_count"), 4);
    writer.set_u32(format!("{arch}.nextn_predict_layers"), 0);
    let tensors: Vec<_> = source.tensors().iter().filter(|t| {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            rest.split('.').next().and_then(|n| n.parse::<usize>().ok()).is_some_and(|l| l < 4)
        } else { t.name.starts_with("token_embd.") || t.name.starts_with("output") }
    }).collect();
    for t in &tensors { writer.add_tensor(&t.name, t.dims.clone(), t.ty)?; }
    writer.write_header()?;
    for t in tensors { writer.write_tensor_data(&t.name, source.tensor_bytes(t)?)?; }
    writer.finish()?;
    drop(source);
    let opts = LoadOptions { ctx: 128, batch: 4, allow_fallback: false, ..Default::default() };
    let (mut resident, _) = loader::load(&path, &opts, |_, _, _| {})?;
    let (mut cached, _) = loader::load(&path, &LoadOptions { expert_cache: Some(8), ..opts }, |_, _, _| {})?;
    let tokens = resident.tokenizer.encode("The capital of France is Paris. Water is cold. Rust and CUDA accelerate inference.", false);
    let mut worst = 0.0f32;
    for chunk_size in [1, 4, 1] {
        resident.reset(); cached.reset();
        for chunk in tokens.chunks(chunk_size) {
            let a = resident.forward(chunk)?;
            let b = cached.forward(chunk)?;
            assert!(a.iter().chain(&b).all(|x| x.is_finite()));
            let error = a.iter().zip(&b).map(|(x,y)| (x-y).abs()).fold(0.0f32, f32::max);
            worst = worst.max(error);
            assert!(error <= 1e-4, "offload changed logits by {error}");
        }
    }
    let (hits, misses) = cached.expert_cache_stats();
    assert!(hits > 0 && misses > 4 * 8 * 3, "must exercise reuse and evictions");
    println!("real-weight offload: max logit error={worst}, hits={hits}, misses={misses}");
    Ok(())
}
