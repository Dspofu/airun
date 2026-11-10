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

    let Some(cuda) = find_cuda(&archs) else {
        println!("cargo:rustc-cfg=airun_no_cuda");
        println!(
            "cargo:warning=No CUDA toolkit found (set CUDA_PATH). Building a CPU-only \
             airun; GPU commands will report that the build lacks kernels."
        );
        return;
    };
    let nvcc = cuda.join("bin").join(exe("nvcc"));

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
            // The same C runtime Rust links: the DLL one (/MD) unless the
            // build asks for crt-static. nvcc's default, the static one,
            // made the linker warn LNK4098 (LIBCMT conflicts) and put two
            // C runtimes in the binary.
            let crt_static = std::env::var("CARGO_CFG_TARGET_FEATURE")
                .is_ok_and(|f| f.split(',').any(|x| x == "crt-static"));
            cmd.args(["-Xcompiler", if crt_static { "/MT" } else { "/MD" }]);
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

    // Each .cu is compiled to its own object in parallel, then the objects are
    // archived. One nvcc call with every source compiles them one after the
    // other on a single core, which kept the build parked at the airun-cuda
    // step for minutes.
    let mut gencode: Vec<String> = archs
        .iter()
        .map(|a| format!("-gencode=arch=compute_{a},code=sm_{a}"))
        .collect();
    // A PTX copy of the newest target keeps future architectures loadable via JIT.
    if let Some(newest) = archs.last() {
        gencode.push(format!("-gencode=arch=compute_{newest},code=compute_{newest}"));
    }

    let mut jobs: Vec<(PathBuf, PathBuf, Command)> = Vec::new();
    for src in &sources {
        let obj = out_dir.join(object(src));
        let mut cmd = base();
        cmd.arg("-c").args(&gencode).arg("-o").arg(&obj).arg(src);
        jobs.push((src.clone(), obj, cmd));
    }

    // Blackwell's FP4 tensor-core instructions exist only in the arch-specific
    // target, so their unit is built for compute_120a alone. Without sm_120 in
    // the list it is built empty and reports itself unsupported.
    let fp4_src = kernels_dir.join("sm120a").join("fp4mma.cu");
    let fp4_obj = out_dir.join(format!("sm120a_{}", object(&fp4_src)));
    let mut fp4 = base();
    fp4.arg("-c");
    if archs.iter().any(|a| a == "120") {
        fp4.arg("-gencode=arch=compute_120a,code=sm_120a");
        fp4.arg("-DAIRUN_FP4_BUILT=1");
    } else {
        let a = &archs[0];
        fp4.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        fp4.arg("-DAIRUN_FP4_BUILT=0");
    }
    fp4.arg("-o").arg(&fp4_obj).arg(&fp4_src);
    jobs.push((fp4_src, fp4_obj.clone(), fp4));

    let objects: Vec<PathBuf> = jobs.iter().map(|(_, obj, _)| obj.clone()).collect();
    compile_parallel(jobs, &nvcc);

    let (fp4_objs, kernel_objs): (Vec<PathBuf>, Vec<PathBuf>) =
        objects.into_iter().partition(|o| *o == fp4_obj);
    archive(base(), &out_dir.join(static_lib("airun_kernels")), &kernel_objs, &nvcc);
    archive(base(), &out_dir.join(static_lib("airun_fp4")), &fp4_objs, &nvcc);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=airun_kernels");
    println!("cargo:rustc-link-lib=static=airun_fp4");

    // Where cudart lives: lib/x64 on Windows; lib64, or the targets/ tree of
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

/// The toolkit to build with: the first candidate (CUDA_PATH and friends, then
/// nvcc on PATH, then the default install folders, newest first) whose nvcc
/// knows every requested architecture.
///
/// Installing CUDA sets CUDA_PATH to the version just installed, so a PC that
/// got an older toolkit after a newer one (12.0 after 13.4) pointed the build
/// at an nvcc without sm_120 and every kernel failed with "Unsupported gpu
/// architecture 'compute_120'", with the right toolkit sitting next to it.
fn find_cuda(archs: &[String]) -> Option<PathBuf> {
    let candidates = cuda_candidates();
    let first = candidates.first()?.clone();
    for root in &candidates {
        let missing = missing_archs(&root.join("bin").join(exe("nvcc")), archs);
        if missing.is_empty() {
            if *root != first {
                println!(
                    "cargo:warning=CUDA at {} cannot build sm_{}; using {} instead. Point CUDA_PATH \
                     at it to silence this",
                    first.display(),
                    missing_archs(&first.join("bin").join(exe("nvcc")), archs).join(", sm_"),
                    root.display()
                );
            }
            return Some(root.clone());
        }
    }
    let missing = missing_archs(&first.join("bin").join(exe("nvcc")), archs);
    panic!(
        "no installed CUDA toolkit can build sm_{} (tried {}). Install CUDA 12.8 or newer for RTX 50 \
         (sm_120), or set AIRUN_CUDA_ARCHS to the architectures of your GPUs (e.g. \"86\")",
        missing.join(", sm_"),
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    );
}

/// Architectures in `archs` that this nvcc cannot emit. An nvcc too old for
/// `--list-gpu-arch` (before CUDA 11) is taken at its word: the build fails
/// with nvcc's own message if it lacks one.
fn missing_archs(nvcc: &Path, archs: &[String]) -> Vec<String> {
    let Ok(out) = Command::new(nvcc).arg("--list-gpu-arch").output() else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let known = String::from_utf8_lossy(&out.stdout).into_owned();
    let known: Vec<&str> = known.split_whitespace().filter_map(|s| s.strip_prefix("compute_")).collect();
    archs.iter().filter(|a| !known.contains(&a.as_str())).cloned().collect()
}

/// Every CUDA toolkit root with an nvcc, in order of preference, without repeats.
fn cuda_candidates() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut push = |p: PathBuf| {
        if p.join("bin").join(exe("nvcc")).exists() && !found.iter().any(|f| same_dir(f, &p)) {
            found.push(p);
        }
    };

    for var in ["CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"] {
        if let Ok(p) = env::var(var) {
            push(PathBuf::from(p));
        }
    }

    // nvcc on PATH: walk back up from bin/.
    if let Ok(path) = env::var("PATH") {
        for dir in env::split_paths(&path) {
            if dir.join(exe("nvcc")).exists() {
                if let Some(root) = dir.parent() {
                    push(root.to_path_buf());
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
        push(root.clone());
        let Ok(entries) = std::fs::read_dir(&root) else {
            continue;
        };
        let mut versions: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        // By number: a string sort puts v12.10 before v12.9.
        versions.sort_by_key(|p| std::cmp::Reverse(version_of(p)));
        for v in versions {
            push(v);
        }
    }
    // Linux installs side by side as /usr/local/cuda-12.8, /usr/local/cuda-13.0...
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut versions: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("cuda-")))
            .collect();
        versions.sort_by_key(|p| std::cmp::Reverse(version_of(p)));
        for v in versions {
            push(v);
        }
    }
    found
}

/// "v13.4" -> [13, 4]; folders without a version sort last.
fn version_of(p: &Path) -> Vec<u32> {
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    name.trim_start_matches(|c: char| !c.is_ascii_digit())
        .split('.')
        .map_while(|s| s.parse().ok())
        .collect()
}

/// Paths from env vars and from the folder listing differ in case and
/// trailing separators on Windows.
fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Object file name for a source: `ops.cu` -> `ops.obj` / `ops.o`.
fn object(src: &Path) -> String {
    let stem = src.file_stem().unwrap().to_string_lossy();
    if cfg!(windows) { format!("{stem}.obj") } else { format!("{stem}.o") }
}

/// How many nvcc processes to run at once. Cargo's `NUM_JOBS` honours `-j`;
/// the cap keeps peak RAM sane, since each cicc of the big units takes ~1-2 GB.
fn parallel_jobs() -> usize {
    let n = env::var("NUM_JOBS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    n.clamp(1, 8)
}

/// Compiles every job, a few at a time, largest sources first so the longest
/// unit starts immediately. All failures are reported, not just the first.
fn compile_parallel(mut jobs: Vec<(PathBuf, PathBuf, Command)>, nvcc: &Path) {
    jobs.sort_by_key(|(src, _, _)| std::cmp::Reverse(std::fs::metadata(src).map(|m| m.len()).unwrap_or(0)));
    let queue = std::sync::Mutex::new(jobs.into_iter());
    let failed = std::sync::Mutex::new(Vec::<String>::new());
    let workers = parallel_jobs();
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let Some((src, _obj, cmd)) = queue.lock().unwrap().next() else { break };
                let name = src.file_name().unwrap().to_string_lossy().into_owned();
                let start = std::time::Instant::now();
                match run_nvcc(cmd, nvcc) {
                    Ok(()) => println!("nvcc {name}: {:.1}s", start.elapsed().as_secs_f32()),
                    Err(log) => {
                        for line in log.lines() {
                            println!("cargo:warning=nvcc {name}: {line}");
                        }
                        failed.lock().unwrap().push(name);
                    }
                }
            });
        }
    });
    let failed = failed.into_inner().unwrap();
    if !failed.is_empty() {
        panic!(
            "nvcc failed on {}. The compiler messages are in the warnings above; \
             fix the kernel and build again",
            failed.join(", ")
        );
    }
}

/// Packs compiled objects into a static library (nvcc drives lib.exe / ar).
fn archive(mut cmd: Command, lib: &Path, objects: &[PathBuf], nvcc: &Path) {
    cmd.arg("-lib").arg("-o").arg(lib).args(objects);
    if let Err(log) = run_nvcc(cmd, nvcc) {
        for line in log.lines() {
            println!("cargo:warning=nvcc -lib: {line}");
        }
        panic!("nvcc could not create {}; see the warnings above", lib.display());
    }
}

/// Runs nvcc, returning its diagnostics verbatim when it fails: they are the
/// single most useful thing when a kernel breaks, more than "build failed".
fn run_nvcc(mut cmd: Command, nvcc: &Path) -> Result<(), String> {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!(
            "failed to run nvcc at {}: {e}. Check that the CUDA Toolkit is installed \
             and CUDA_PATH points at it",
            nvcc.display()
        );
    });
    if output.status.success() {
        return Ok(());
    }
    let mut log = String::from_utf8_lossy(&output.stderr).into_owned();
    log.push_str(&String::from_utf8_lossy(&output.stdout));
    log.push_str(&format!("(exit status {})", output.status));
    Err(log)
}
