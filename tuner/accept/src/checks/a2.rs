//! A2: every fixture under fixtures/decide/ through `tuner decide`, black box: the verdict and
//! reason kind (and detail substring when the fixture names one), and one delta per workload
//! series matching a recompute from the fixture's samples.

use std::fs;
use std::time::Duration;

use tuner::contract::{exit, DecideFixture, DecideOutput, Thresholds, SCHEMA_VERSION};

use super::{describe, fail, pass, Ctx};
use crate::rules::results::delta_pct;
use crate::rules::{compare_decision, series_stats};
use crate::status::{CheckResult, Evidence};

fn check_deltas(f: &DecideFixture, d: &DecideOutput) -> Result<(), String> {
    let want: Vec<(String, f64)> = f
        .workload
        .series()
        .into_iter()
        .filter_map(|s| {
            let (b, c) = (series_stats(&f.baseline.samples, s)?, series_stats(&f.candidate.samples, s)?);
            Some((s.to_string(), delta_pct(&b, &c)))
        })
        .collect();
    let got: Vec<(String, f64)> = d.deltas.iter().map(|x| (x.series.to_string(), x.delta_pct.0)).collect();
    let same = got.len() == want.len() && got.iter().zip(&want).all(|(g, w)| g.0 == w.0 && (g.1 - w.1).abs() <= 1e-6 * w.1.abs().max(1.0));
    if same {
        Ok(())
    } else {
        Err(format!("deltas {got:?} != recomputed {want:?}"))
    }
}

pub fn check(ctx: &Ctx) -> CheckResult {
    let dir = ctx.fixture("decide");
    let mut paths: Vec<_> = match fs::read_dir(&dir) {
        Ok(rd) => rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect(),
        Err(e) => return fail(format!("{}: {e}", dir.display()), vec![]),
    };
    paths.sort();
    if paths.is_empty() {
        return fail(format!("no fixtures in {}", dir.display()), vec![]);
    }
    let mut evidence = Vec::new();
    let mut failures = Vec::new();
    for path in &paths {
        let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let fixture: DecideFixture = match fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
            Ok(f) => f,
            Err(e) => {
                failures.push(format!("{name}: fixture does not parse as DecideFixture: {e}"));
                continue;
            }
        };
        if fixture.schema_version != SCHEMA_VERSION || fixture.thresholds != Thresholds::default() {
            failures.push(format!("{name}: fixture thresholds/schema drift from the contract defaults"));
            continue;
        }
        let path_s = path.to_string_lossy().into_owned();
        let args = ["decide", "--fixture", &path_s];
        let out = match ctx.tuner(&args, Duration::from_secs(60), Duration::ZERO) {
            Ok(o) => o,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        evidence.push(Evidence::new(describe(&ctx.tuner_bin, &args), &out.combined()));
        if out.status != Some(exit::OK) {
            failures.push(format!("{name}: exit {:?}", out.status));
            continue;
        }
        match serde_json::from_str::<DecideOutput>(&out.stdout) {
            Ok(d) if d.schema_version == SCHEMA_VERSION => {
                if let Err(e) = compare_decision(&fixture.expect, &d.decision).and_then(|()| check_deltas(&fixture, &d)) {
                    failures.push(format!("{name}: {e}"));
                }
            }
            Ok(d) => failures.push(format!("{name}: schema_version {}", d.schema_version)),
            Err(e) => failures.push(format!("{name}: stdout is not a DecideOutput: {e}")),
        }
    }
    if failures.is_empty() {
        pass(format!("{} fixtures decided as expected", paths.len()), evidence)
    } else {
        fail(format!("{}/{} fixtures wrong: {}", failures.len(), paths.len(), failures.join(" | ")), evidence)
    }
}
