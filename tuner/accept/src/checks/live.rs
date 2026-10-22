//! The part of a live check (`--live`, strix only) that is the same for every spec: gate on the
//! harness's own preflight, run `tuner sweep`, audit for orphans, and hand back the outputs to judge.

use std::path::PathBuf;
use std::time::Duration;

use tuner::contract::exit;

use super::{describe, fail, skip, Ctx, strix_gguf};
use crate::remote::{host_state, llama_processes, logits_files_since, LLAMA_PGREP};
use crate::rules::{live_gate, orphans, GIB};
use crate::status::{CheckResult, Evidence};

pub const HOST: &str = "strix";
/// The tuner gets TERM and this long to reap before KILL.
const REAP_GRACE: Duration = Duration::from_secs(90);
/// Headroom on strix's disk the live checks demand beyond the tuner's own estimate, so a
/// mis-sized logits file cannot fill the filesystem llama-swap serves from.
const DISK_SLACK_GIB: u64 = 50;
/// gpt-oss (n_vocab 201088) base logits: 8192 scored tokens x 2 bytes at `kld_ctx` 512, 32768 at 32768
/// (contract "Preflight"; the casebook measured 13.2 GB for the latter).
pub const LOGITS_BYTES_512: u64 = 201_088 * 2 * 8192;
pub const LOGITS_BYTES_32K: u64 = 201_088 * 2 * 32768;

pub struct Sweep {
    pub out_dir: PathBuf,
    pub recipes: PathBuf,
    /// RFC 3339 UTC just before the tuner started; the recipe must be newer.
    pub started: String,
    pub evidence: Vec<Evidence>,
}

/// Gate, run `tuner sweep --spec <spec>` with `timeout`, audit. `Err` is the check's result
/// (skip when strix is busy or unreachable, fail on a leak, a timeout, or a non-zero exit).
/// `logits_bytes` is the base logits file the spec's `kld_ctx` implies (contract "Preflight"); the
/// harness skips when strix's disk cannot hold it with `DISK_SLACK_GIB` to spare.
pub fn sweep(ctx: &Ctx, tag: &str, spec: &str, timeout: Duration, logits_bytes: u64) -> Result<Sweep, CheckResult> {
    let mut evidence = Vec::new();
    let state = host_state(HOST, strix_gguf()).map_err(|e| skip(format!("strix unreachable: {e}"), vec![Evidence::new("ssh strix (state)", &e)]))?;
    evidence.push(Evidence::new(
        "ssh strix: MemAvailable, :8089/running, llama procs, model size, free disk",
        &format!(
            "MemAvailable {:.1} GiB; running {:?}; procs {:?}; model {:?} bytes; disk_avail {:.1} GiB",
            state.mem_available as f64 / GIB as f64,
            state.running,
            state.procs,
            state.model_bytes,
            state.disk_avail as f64 / GIB as f64
        ),
    ));
    let Some(model_bytes) = state.model_bytes else {
        return Err(skip(format!("{} not on strix", strix_gguf()), evidence));
    };
    if let Err(reason) = live_gate(state.mem_available, model_bytes, &state.running, &state.procs) {
        return Err(skip(format!("strix busy: {reason}"), evidence));
    }
    if state.disk_avail < logits_bytes + DISK_SLACK_GIB * GIB {
        return Err(skip(
            format!("strix disk: {:.1} GiB free < logits {:.1} GiB + {DISK_SLACK_GIB} GiB slack", state.disk_avail as f64 / GIB as f64, logits_bytes as f64 / GIB as f64),
            evidence,
        ));
    }

    let out_dir = ctx.out.join(format!("{tag}-{}", ctx.run_id));
    let recipes = ctx.out.join(format!("recipes-{tag}-{}", ctx.run_id));
    let (out_s, rec_s) = (out_dir.to_string_lossy().into_owned(), recipes.to_string_lossy().into_owned());
    let spec = ctx.fixture(spec).to_string_lossy().into_owned();
    let args = ["sweep", "--host", HOST, "--model", strix_gguf(), "--spec", &spec, "--out", &out_s, "--recipes", &rec_s];
    let started = crate::utc_now();
    let out = ctx.tuner(&args, timeout, REAP_GRACE).map_err(|e| fail(e.clone(), evidence.clone()))?;
    evidence.push(Evidence::new(describe(&ctx.tuner_bin, &args), &out.combined()));

    // Orphan audit first: a leak is a failure whatever the numbers say.
    match llama_processes(HOST) {
        Ok(after) => {
            let left = orphans(&state.procs, &after);
            evidence.push(Evidence::new(format!("ssh strix {LLAMA_PGREP} (after)"), if after.is_empty() { "none" } else { &after }));
            if !left.is_empty() {
                return Err(fail(format!("orphan llama.cpp process(es) after the sweep: {}", left.join("; ")), evidence));
            }
        }
        Err(e) => return Err(fail(format!("orphan audit failed: {e}"), evidence)),
    }
    // The base logits file (GBs; 13 GB at 32k) must be gone with the run, whatever the exit.
    match logits_files_since(HOST, &started) {
        Ok(left) => {
            evidence.push(Evidence::new(format!("ssh strix find ~/.tuner/runs -name '*.kld' -newermt {started}"), if left.is_empty() { "none" } else { &left }));
            if !left.is_empty() {
                return Err(fail(format!("logits file(s) left on strix after the sweep: {}", left.replace('\n', "; ")), evidence));
            }
        }
        Err(e) => return Err(fail(format!("logits audit failed: {e}"), evidence)),
    }
    if out.timed_out {
        return Err(fail(format!("sweep exceeded {} min", timeout.as_secs() / 60), evidence));
    }
    if out.status != Some(exit::OK) {
        return Err(fail(format!("sweep exited {:?}: {}", out.status, out.stderr.trim().lines().last().unwrap_or("")), evidence));
    }
    Ok(Sweep { out_dir, recipes, started, evidence })
}
