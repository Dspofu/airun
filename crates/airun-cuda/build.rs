//! Compiles the CUDA kernels in `kernels/` into a static library and links it.
//!
//! nvcc needs a host compiler, and on Windows that means finding MSVC without a
//! Developer Prompt. The `cc` crate already knows how to locate the toolchain,
//! so we borrow its answer and hand nvcc a `-ccbin` plus the INCLUDE/LIB
//! environment it expects.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Architectures we emit SASS for by default.
///
/// The task calls for RTX 20-series and newer, which is exactly compute 7.5
/// upward. We stop at the newest real target and add a PTX copy of it so future
/// cards still JIT.
const DEFAULT_ARCHS: &[&str] = &["75", "80", "86", "89", "120"];

fn main() {
    // Declared so `--cfg airun_cuda` does not trip the unexpected_cfgs lint.
    println!("cargo::rustc-check-cfg=cfg(airun_cuda)");
    println!("cargo::rustc-check-cfg=cfg(airun_no_cuda)");

    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AIRUN_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=AIRUN_NVCC_CCBIN");
    println!("cargo:rerun-if-env-changed=AIRUN_SKIP_CUDA");

    // Lets `cargo check`/docs.rs style flows work on machines with no CUDA.
    if env::var("AIRUN_SKIP_CUDA").is_ok() {
        println!("cargo:rustc-cfg=airun_no_cuda");
        println!("cargo:warning=AIRUN_SKIP_CUDA set: building without GPU kernels");
        return;
    }

    let cuda = match find_cuda() {
        Some(p) => p,
        None => {
            println!("cargo:rustc-cfg=airun_no_cuda");
            println!(
                "cargo:warning=No CUDA toolkit found (set CUDA_PATH). Building a CPU-only \
                 airun; GPU commands will report that the build lacks kernels."
            );
            return;
        }
    };

    let nvcc = cuda.join("bin").join(exe("nvcc"));
    if !nvcc.exists() {
        println!("cargo:rustc-cfg=airun_no_cuda");
        println!("cargo:warning=CUDA_PATH={} has no nvcc", cuda.display());
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let kernels_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("kernels");

    let mut sources: Vec<PathBuf> = std::fs::read_dir(&kernels_dir)
        .expect("kernels/ directory is missing")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "cu"))
        .collect();
    sources.sort();
    assert!(!sources.is_empty(), "no .cu files in {}", kernels_dir.display());

    let archs: Vec<String> = env::var("AIRUN_CUDA_ARCHS")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.split(&[',', ' '][..])
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().to_string())
                .collect()
        })
        .unwrap_or_else(|| DEFAULT_ARCHS.iter().map(|s| s.to_string()).collect());

    // Flags, include path, host compiler and its environment, shared by both
    // libraries below.
    let base = || {
        let mut cmd = Command::new(&nvcc);
        cmd.args([
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--extended-lambda",
            // Host compilers release faster than CUDA's supported-compiler table;
            // the C++ we generate is plain enough that the check is not worth a
            // hard stop.
            "-allow-unsupported-compiler",
            "-diag-suppress=177", // unused function: kernels are selected per type
        ]);
        if cfg!(windows) {
            // MSVC's C++ exception model (gcc and clang need no flag).
            cmd.args(["-Xcompiler", "/EHsc"]);
        } else {
            // Static libraries linked into a Rust binary, which is position
            // independent by default on Linux.
            cmd.args(["-Xcompiler", "-fPIC"]);
        }
        cmd.arg("-I").arg(&kernels_dir);
        if let Some(ccbin) = host_compiler_dir() {
            cmd.arg("-ccbin").arg(&ccbin);
        }
        // The MSVC environment (INCLUDE/LIB) has to reach the host compiler nvcc spawns.
        if let Some(tool) = cc::windows_registry::find_tool(&target(), "cl.exe") {
            for (k, v) in tool.env() {
                cmd.env(k, v);
            }
        }
        cmd
    };

    let mut cmd = base();
    cmd.arg("-lib");
    cmd.arg("-o").arg(out_dir.join(static_lib("airun_kernels")));
    for arch in &archs {
        cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    // A PTX copy of the newest target keeps future architectures loadable via JIT.
    if let Some(newest) = archs.last() {
        cmd.arg(format!("-gencode=arch=compute_{newest},code=compute_{newest}"));
    }
    for src in &sources {
        cmd.arg(src);
    }
    run_nvcc(cmd, &nvcc);

    // Blackwell's FP4 tensor-core instructions exist only in the arch-specific
    // target, so their unit is built for compute_120a alone. Without sm_120 in
    // the list it is built empty and reports itself unsupported.
    let mut fp4 = base();
    fp4.arg("-lib");
    fp4.arg("-o").arg(out_dir.join(static_lib("airun_fp4")));
    if archs.iter().any(|a| a == "120") {
        fp4.arg("-gencode=arch=compute_120a,code=sm_120a");
        fp4.arg("-DAIRUN_FP4_BUILT=1");
    } else {
        let a = &archs[0];
        fp4.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        fp4.arg("-DAIRUN_FP4_BUILT=0");
    }
    fp4.arg(kernels_dir.join("sm120a").join("fp4mma.cu"));
    run_nvcc(fp4, &nvcc);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=airun_kernels");
    println!("cargo:rustc-link-lib=static=airun_fp4");

    // Where cudart lives: libd on Windows; lib64, or the targets/ tree of
    // a distribution package, on Linux.
    for dir in [cuda.join("lib").join("x64"), cuda.join("lib64"), cuda.join("targets").join("x86_64-linux").join("lib"),
        cuda.join("lib")] {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
        }
    }
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !cfg!(windows) {
        // The kernels' host code is C++; MSVC links its runtime on its own.
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    println!("cargo:rustc-cfg=airun_cuda");
    println!("cargo:cuda_path={}", cuda.display());
    println!("cargo:archs={}", archs.join(","));
}

/// A static library's file name as the platform's linker looks for it.
fn static_lib(name: &str) -> String {
    if cfg!(windows) { format!("{name}.lib") } else { format!("lib{name}.a") }
}

fn target() -> String {
    env::var("TARGET").unwrap_or_else(|_| "x86_64-pc-windows-msvc".into())
}

fn exe(name: &str) -> String {
    if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    }
}

/// Directory containing the host compiler nvcc should drive.
fn host_compiler_dir() -> Option<PathBuf> {
    if let Ok(explicit) = env::var("AIRUN_NVCC_CCBIN") {
        if !explicit.trim().is_empty() {
            return Some(PathBuf::from(explicit));
        }
    }
    if !cfg!(windows) {
        return None;
    }
    let tool = cc::windows_registry::find_tool(&target(), "cl.exe")?;
    tool.path().parent().map(Path::to_path_buf)
}

fn find_cuda() -> Option<PathBuf> {
    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(p) = env::var(var) {
            let p = PathBuf::from(p);
            if p.join("bin").join(exe("nvcc")).exists() {
                return Some(p);
            }
        }
    }

    // nvcc on PATH: walk back up from bin/.
    if let Ok(path) = env::var("PATH") {
        for dir in env::split_paths(&path) {
            if dir.join(exe("nvcc")).exists() {
                if let Some(root) = dir.parent() {
                    return Some(root.to_path_buf());
                }
            }
        }
    }

    // Default install locations, newest version first.
    let roots = [
        PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        PathBuf::from("/usr/local/cuda"),
    ];
    for root in roots {
        if root.join("bin").join(exe("nvcc")).exists() {
            return Some(root);
        }
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.join("bin").join(exe("nvcc")).exists())
            .collect();
        versions.sort();
        if let Some(newest) = versions.pop() {
            return Some(newest);
        }
    }
    None
}

/// Runs nvcc, surfacing its diagnostics verbatim when it fails: they are the
/// single most useful thing when a kernel breaks, more than "build failed".
fn run_nvcc(mut cmd: Command, nvcc: &Path) {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!("failed to run nvcc at {}: {e}", nvcc.display());
    });
    if !output.status.success() {
        for line in String::from_utf8_lossy(&output.stderr).lines() {
            println!("cargo:warning=nvcc: {line}");
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            println!("cargo:warning=nvcc: {line}");
        }
        panic!("nvcc failed with status {}", output.status);
    }
}
