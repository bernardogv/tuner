//! A4: preflight refusals. (a) strix with an impossible reserve; (b) a Spark without the
//! spark-mode acknowledgement. Both must exit `REFUSED` with nothing started on the host.

use std::time::Duration;

use tuner::contract::exit;

use super::{describe, fail, pass, skip, Ctx, strix_gguf};
use crate::remote::{llama_processes, LLAMA_PGREP};
use crate::rules::orphans;
use crate::status::{CheckResult, Evidence};

const SPARK_HOST: &str = "spark2";
/// Any path: the tuner must refuse before it looks at the model.
const SPARK_GGUF: &str = "/models/never-loaded.gguf";

struct Refusal {
    out: crate::proc::Captured,
    orphans: Vec<String>,
    results_written: bool,
}

fn refusal(ctx: &Ctx, host: &str, args: &[&str], out_dir: &std::path::Path, evidence: &mut Vec<Evidence>) -> Result<Refusal, CheckResult> {
    let before = llama_processes(host).map_err(|e| skip(format!("cannot audit {host}: {e}"), vec![Evidence::new(format!("ssh {host} {LLAMA_PGREP}"), &e)]))?;
    evidence.push(Evidence::new(format!("ssh {host} {LLAMA_PGREP} (before)"), if before.is_empty() { "none" } else { &before }));
    let out = ctx.tuner(args, Duration::from_secs(300), Duration::from_secs(30)).map_err(|e| fail(e.clone(), vec![Evidence::new("tuner", &e)]))?;
    evidence.push(Evidence::new(describe(&ctx.tuner_bin, args), &out.combined()));
    let after = llama_processes(host).map_err(|e| fail(format!("audit after the run failed on {host}: {e}"), evidence.clone()))?;
    evidence.push(Evidence::new(format!("ssh {host} {LLAMA_PGREP} (after)"), if after.is_empty() { "none" } else { &after }));
    Ok(Refusal { out, orphans: orphans(&before, &after), results_written: out_dir.join("results.json").exists() })
}

fn judge(r: &Refusal, must_mention: &[&str], evidence: Vec<Evidence>) -> CheckResult {
    if !r.orphans.is_empty() {
        return fail(format!("llama.cpp process(es) appeared on the host: {}", r.orphans.join("; ")), evidence);
    }
    if r.results_written {
        return fail("results.json was written despite the refusal", evidence);
    }
    if r.out.status != Some(exit::REFUSED) {
        return fail(format!("expected exit {} (refused), got {:?}: {}", exit::REFUSED, r.out.status, r.out.stderr.trim()), evidence);
    }
    let err = r.out.stderr.to_lowercase();
    if !must_mention.iter().any(|m| err.contains(&m.to_lowercase())) {
        return fail(format!("refusal does not name its reason (expected one of {must_mention:?}): {}", r.out.stderr.trim()), evidence);
    }
    pass(format!("refused with exit {}: {}", exit::REFUSED, r.out.stderr.trim().lines().next().unwrap_or("")), evidence)
}

pub fn strix(ctx: &Ctx) -> CheckResult {
    let out_dir = ctx.out.join(format!("a4-strix-{}", ctx.run_id));
    let out_s = out_dir.to_string_lossy().into_owned();
    let spec = ctx.fixture("live.toml").to_string_lossy().into_owned();
    let args = ["sweep", "--host", "strix", "--model", strix_gguf(), "--spec", &spec, "--out", &out_s, "--reserve-gib", "100000"];
    let mut evidence = Vec::new();
    match refusal(ctx, "strix", &args, &out_dir, &mut evidence) {
        // Residency is refused before the reserve is looked at (contract "Preflight" order), so a
        // busy strix makes the reserve rule unobservable: a skip, like the live checks.
        Ok(r) if r.out.status == Some(exit::REFUSED) && r.out.stderr.contains("never load beside a resident model") => {
            skip(format!("strix busy, reserve rule unobservable: {}", r.out.stderr.trim()), evidence)
        }
        // Presence is refused first too: without the model the reserve rule is equally unobservable.
        Ok(r) if r.out.status == Some(exit::REFUSED) && r.out.stderr.contains("not found on the host") => {
            skip(format!("set TUNER_GGUF to the model's path on strix: {}", r.out.stderr.trim()), evidence)
        }
        Ok(r) => judge(&r, &["reserve", "memavailable", "headroom"], evidence),
        Err(r) => r,
    }
}

pub fn spark(ctx: &Ctx) -> CheckResult {
    let out_dir = ctx.out.join(format!("a4-spark-{}", ctx.run_id));
    let out_s = out_dir.to_string_lossy().into_owned();
    let spec = ctx.fixture("live.toml").to_string_lossy().into_owned();
    let args = ["sweep", "--host", SPARK_HOST, "--model", SPARK_GGUF, "--spec", &spec, "--out", &out_s];
    let mut evidence = Vec::new();
    match refusal(ctx, SPARK_HOST, &args, &out_dir, &mut evidence) {
        Ok(r) => judge(&r, &["spark-mode"], evidence),
        Err(r) => r,
    }
}
