//! A8 (`--live --kld-32k`, strix only): KLD at depth. fixtures/live-kld32k.toml benches depth 0
//! only (cheap) but measures quality with `kld_ctx = 32768`: baseline logits, the reorder-only
//! control and one candidate (f16 K / q8_0 V, the casebook's conservative pick) each score 32k
//! tokens at 16k-32k depth. Asserts the logs show `-c 32768 --chunks 2` ran, the control is under
//! the cap, every verdict matches the relative rule, and the 13 GB logits file is gone (live.rs).
//! Costs ~20 min and ~13 GB of disk on strix, hence its own flag.

use std::fs;
use std::path::Path;
use std::time::Duration;

use tuner::contract::{Spec, Thresholds, Verdict};

use super::a5::{judge_recipe_and_casebook, read_results};
use super::{fail, live, pass, skip, Ctx};
use crate::status::CheckResult;

const SPEC: &str = "live-kld32k.toml";
/// Three 32k KLD runs (~5 min each measured) plus two depth-0 benches and model loads: ~20 min.
const SWEEP_TIMEOUT: Duration = Duration::from_secs(60 * 60);

fn judge_outputs(spec: &Spec, out_dir: &Path, recipes: &Path, started: &str) -> Result<String, String> {
    let results = read_results(out_dir)?;
    let w = &spec.workload;
    for name in ["kld-0.log", "kld-control.log", "kld-1.log"] {
        let log = fs::read_to_string(out_dir.join(name)).map_err(|e| format!("{name}: {e}"))?;
        if !log.contains("n_ctx=32768") || !log.contains("over 2 chunks") {
            return Err(format!("{name} does not show a `-c 32768 --chunks 2` run (expected `calculating perplexity over 2 chunks, n_ctx=32768`)"));
        }
    }
    let t = Thresholds::default();
    let ctl = results.control.kld;
    if ctl.mean_kld.0 > t.control_kld_cap.0 {
        return Err(format!("control KLD {:.4} at 32k is over the cap {:.4}: the instrument, not the candidate, is broken", ctl.mean_kld.0, t.control_kld_cap.0));
    }
    let mut lines = Vec::new();
    for (i, p) in results.points.iter().enumerate() {
        let expected = crate::rules::expected_kind(w, &t, spec.repeats, ctl, &results.baseline, p);
        let got = p.decision.reason.kind;
        if got != expected {
            return Err(format!("points[{i}] {:?}: tuner says {got:?} ({}), the measured numbers say {expected:?}", p.config, p.decision.reason.detail));
        }
        let q = p.kld.map_or("no KLD".to_string(), |q| format!("KLD {:.4} / {:.1}%", q.mean_kld.0, q.same_top_pct.0));
        lines.push(format!("{:?}/{:?} -> {:?}/{got:?} [{q}]", p.config.ctk, p.config.ctv, p.decision.verdict));
    }
    judge_recipe_and_casebook(&results, out_dir, recipes, started, w.objective, w.kld_ctx)?;
    let kept = results.points.iter().filter(|p| p.decision.verdict == Verdict::Keep).count();
    Ok(format!(
        "kld_ctx 32768: control ub {} KLD {:.4} / {:.1}%; {}; {kept} kept",
        results.control.config.ub,
        ctl.mean_kld.0,
        ctl.same_top_pct.0,
        lines.join("; ")
    ))
}

pub fn check(ctx: &Ctx) -> CheckResult {
    if !(ctx.live && ctx.kld_32k) {
        return skip("pass --live --kld-32k to run the 32k-context KLD sweep on strix (~20 min, ~13 GB disk)", vec![]);
    }
    let spec_path = ctx.fixture(SPEC);
    let spec: Spec = match fs::read_to_string(&spec_path).map_err(|e| e.to_string()).and_then(|t| toml::from_str(&t).map_err(|e| e.to_string())) {
        Ok(s) => s,
        Err(e) => return fail(format!("{}: {e}", spec_path.display()), vec![]),
    };
    if spec.workload.kld_ctx != 32768 {
        return fail(format!("{SPEC} must set kld_ctx = 32768, has {}", spec.workload.kld_ctx), vec![]);
    }
    let run = match live::sweep(ctx, "a8", SPEC, SWEEP_TIMEOUT, live::LOGITS_BYTES_32K) {
        Ok(r) => r,
        Err(result) => return result,
    };
    match judge_outputs(&spec, &run.out_dir, &run.recipes, &run.started) {
        Ok(summary) => pass(summary, run.evidence),
        Err(e) => fail(e, run.evidence),
    }
}
