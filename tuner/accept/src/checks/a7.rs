//! A7 (`--live`, strix only): the casebook's HIP-at-32k tradeoff as a real sweep with
//! fixtures/live-depth.toml (depths [0, 32768], objective pp@32768, Vulkan baseline vs HIP).
//! Asserts every point carries all four series and every verdict is what the keep rule says
//! about the measured numbers (recomputed here, `rules::expected_kind`). Which backend wins is
//! not asserted.

use std::fs;
use std::path::Path;
use std::time::Duration;

use tuner::contract::{Depth, Series, Spec, Thresholds, Verdict};

use super::a5::{judge_recipe_and_casebook, read_results};
use super::{fail, live, pass, skip, Ctx};
use crate::rules::{anchor_within, series_stats, ANCHOR_TOLERANCE_PCT, STRIX_GPT_OSS_ANCHOR};
use crate::status::CheckResult;

const SPEC: &str = "live-depth.toml";
/// Two configs x (3 repeats x 2 rows at 32k depth: ~95 s each on Vulkan, ~65 s on HIP) plus two
/// KLD runs and two model loads: ~25 min expected. The per-bench timeout inside the tuner must
/// allow a 32k row set too (BENCH_TIMEOUT was sized for depth 0).
const SWEEP_TIMEOUT: Duration = Duration::from_secs(75 * 60);

fn judge_outputs(spec: &Spec, out_dir: &Path, recipes: &Path, started: &str) -> Result<String, String> {
    let results = read_results(out_dir)?;
    let w = &spec.workload;
    let want: Vec<Series> = w.series();
    for (tag, p) in std::iter::once(("baseline", &results.baseline)).chain(results.points.iter().map(|p| ("candidate", p))) {
        if p.failure.is_none() {
            let got: Vec<Series> = p.series.iter().map(|x| x.series).collect();
            if got != want {
                return Err(format!(
                    "{tag} {:?}: series {:?}, expected {:?}",
                    p.config,
                    got.iter().map(ToString::to_string).collect::<Vec<_>>(),
                    want.iter().map(ToString::to_string).collect::<Vec<_>>()
                ));
            }
        }
    }
    let tg = series_stats(&results.baseline.samples, Series::TG0).ok_or("baseline has no tg@0")?;
    anchor_within(tg.mean, STRIX_GPT_OSS_ANCHOR, ANCHOR_TOLERANCE_PCT)?;

    let t = Thresholds::default();
    let mut lines = Vec::new();
    for (i, p) in results.points.iter().enumerate() {
        let expected = crate::rules::expected_kind(w, &t, spec.repeats, results.control.kld, &results.baseline, p);
        let got = p.decision.reason.kind;
        if got != expected {
            return Err(format!(
                "points[{i}] {:?}: tuner says {got:?} ({}), the measured numbers say {expected:?}",
                p.config, p.decision.reason.detail
            ));
        }
        let deltas: Vec<String> = want
            .iter()
            .filter_map(|s| {
                let (b, c) = (series_stats(&results.baseline.samples, *s)?, series_stats(&p.samples, *s)?);
                Some(format!("{s} {:+.1}%", crate::rules::results::delta_pct(&b, &c)))
            })
            .collect();
        lines.push(format!("{:?}+hipblaslt={} -> {:?}/{got:?} [{}]", p.config.backend, p.config.hipblaslt, p.decision.verdict, deltas.join(", ")));
    }
    let recipe = judge_recipe_and_casebook(&results, out_dir, recipes, started, w.objective, w.kld_ctx)?;
    let kept = results.points.iter().filter(|p| p.decision.verdict == Verdict::Keep).count();
    let pp32k = series_stats(&results.baseline.samples, Series { metric: tuner::contract::Metric::Pp, depth: Depth(32768) }).map_or(0.0, |s| s.mean.0);
    Ok(format!(
        "baseline tg@0 {:.2} / pp@32768 {pp32k:.1} tok/s; control ub {} KLD {:.4} / {:.1}%; {}; {kept} kept, guard {}%, recipe winner {:?} by {}",
        tg.mean.0,
        results.control.config.ub,
        results.control.kld.mean_kld.0,
        results.control.kld.same_top_pct.0,
        lines.join("; "),
        w.guard_pct.0,
        recipe.winner.backend,
        recipe.objective
    ))
}

pub fn check(ctx: &Ctx) -> CheckResult {
    if !ctx.live {
        return skip("pass --live to run the depth sweep on strix (~25 min)", vec![]);
    }
    let spec_path = ctx.fixture(SPEC);
    let spec: Spec = match fs::read_to_string(&spec_path).map_err(|e| e.to_string()).and_then(|t| toml::from_str(&t).map_err(|e| e.to_string())) {
        Ok(s) => s,
        Err(e) => return fail(format!("{}: {e}", spec_path.display()), vec![]),
    };
    if spec.workload.depths != [Depth(0), Depth(32768)] {
        return fail(format!("{SPEC} must measure depths [0, 32768], has {:?}", spec.workload.depths), vec![]);
    }
    let run = match live::sweep(ctx, "a7", SPEC, SWEEP_TIMEOUT, live::LOGITS_BYTES_512) {
        Ok(r) => r,
        Err(result) => return result,
    };
    match judge_outputs(&spec, &run.out_dir, &run.recipes, &run.started) {
        Ok(summary) => pass(summary, run.evidence),
        Err(e) => fail(e, run.evidence),
    }
}
