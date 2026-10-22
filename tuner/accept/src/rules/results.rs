//! What a `results.json` must satisfy beyond parsing, and the harness's own copy of the keep rule
//! (contract.rs "Keep rule") so a live sweep's verdicts can be recomputed from the measured numbers.

use std::collections::BTreeSet;

use tuner::contract::{
    Config, Metric, Point, Quality, ReasonKind, Results, Sample, Series, Stats, Thresholds, TokS, Verdict, Workload,
    SCHEMA_VERSION,
};

/// The harness's copy of the reorder-only control rule (contract.rs "Quality"): `ub*2` while that
/// stays `<= b`, else `ub/2`; `None` when neither differs from the baseline.
pub fn control_config(base: Config) -> Option<Config> {
    let ub = if base.ub.saturating_mul(2) <= base.b { base.ub * 2 } else { base.ub / 2 };
    (ub != base.ub && ub > 0).then_some(Config { ub, ..base })
}

/// Mean and sample stdev (n-1) of one series, recomputed from the samples; `None` when absent.
pub fn series_stats(samples: &[Sample], s: Series) -> Option<Stats> {
    let xs: Vec<f64> = samples
        .iter()
        .filter(|x| x.depth == s.depth)
        .map(|x| match s.metric {
            Metric::Pp => x.pp_tok_s.0,
            Metric::Tg => x.tg_tok_s.0,
        })
        .collect();
    let n = xs.len();
    if n == 0 {
        return None;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = if n > 1 { xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64 } else { 0.0 };
    Some(Stats { mean: TokS(mean), stdev: TokS(var.sqrt()), n: u32::try_from(n).unwrap_or(u32::MAX) })
}

/// `(mean_c - mean_b) / mean_b * 100`.
pub fn delta_pct(b: &Stats, c: &Stats) -> f64 {
    (c.mean.0 - b.mean.0) / b.mean.0 * 100.0
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// 100% is the llama.cpp single-stream anchor the hardware row's decode_bw was measured from, not a
/// physical ceiling: a run a hair faster than the anchor scores >100. The ceiling is
/// spec_bw / decode_bw (gb10 1.46, strix 1.31), so 150 bounds every fleet box.
const ROOFLINE_PCT_MAX: f64 = 150.0;

/// Schema-level rules a `results.json` must satisfy beyond parsing: one `SeriesStats` per metric
/// at every depth that has samples, each matching a recompute; `roofline_pct` iff tg@0; the control
/// is the baseline with only `ub` changed by the rule; a measured candidate without a KLD carries
/// a failure.
pub fn validate_results(r: &Results) -> Vec<String> {
    let mut errs = Vec::new();
    if r.schema_version != SCHEMA_VERSION {
        errs.push(format!("schema_version {} != {SCHEMA_VERSION}", r.schema_version));
    }
    let b = &r.baseline;
    if b.decision.verdict != Verdict::Baseline {
        errs.push(format!("baseline verdict must be baseline, got {:?}", b.decision.verdict));
    }
    if b.kld.is_some() {
        errs.push("baseline must not carry a KLD (it is the reference)".into());
    }
    if b.series.is_empty() {
        errs.push("baseline has no series stats".into());
    }
    match control_config(b.config) {
        Some(want) if r.control.config == want => {}
        want => errs.push(format!("control config {:?} is not the baseline with ub {:?}", r.control.config, want.map(|c| c.ub))),
    }
    if r.roofline.expected_tg_tok_s.0 <= 0.0 {
        errs.push("roofline.expected_tg_tok_s must be > 0".into());
    }
    for (i, p) in std::iter::once(b).chain(r.points.iter()).enumerate() {
        let tag = if i == 0 { "baseline".to_string() } else { format!("points[{}]", i - 1) };
        let depths: BTreeSet<_> = p.samples.iter().map(|s| s.depth).collect();
        for d in &depths {
            let n = p.samples.iter().filter(|s| s.depth == *d).count();
            if n > p.repeats as usize {
                errs.push(format!("{tag}: {n} samples at depth {} > repeats {}", d.0, p.repeats));
            }
        }
        let want: Vec<Series> = [Metric::Pp, Metric::Tg]
            .into_iter()
            .flat_map(|metric| depths.iter().map(move |&depth| Series { metric, depth }))
            .collect();
        let got: Vec<Series> = p.series.iter().map(|x| x.series).collect();
        if got != want {
            errs.push(format!(
                "{tag}: series {:?} != every metric at every sampled depth {:?}",
                got.iter().map(ToString::to_string).collect::<Vec<_>>(),
                want.iter().map(ToString::to_string).collect::<Vec<_>>()
            ));
        }
        for x in &p.series {
            match series_stats(&p.samples, x.series) {
                Some(s) if s.n == x.stats.n && close(s.mean.0, x.stats.mean.0) && close(s.stdev.0, x.stats.stdev.0) => {}
                other => errs.push(format!("{tag}: {} stats {:?} do not match the samples ({other:?})", x.series, x.stats)),
            }
        }
        let tg0 = p.series.iter().find(|x| x.series == Series::TG0).map(|x| x.stats.mean.0);
        match (tg0, &p.roofline_pct) {
            (Some(mean), Some(pct)) => {
                let want = mean / r.roofline.expected_tg_tok_s.0 * 100.0;
                if !close(pct.0, want) {
                    errs.push(format!("{tag}: roofline_pct {} != tg@0 {mean:.3} / expected {:.3} = {want:.3}", pct.0, r.roofline.expected_tg_tok_s.0));
                }
                if !(pct.0 > 0.0 && pct.0 <= ROOFLINE_PCT_MAX) {
                    errs.push(format!("{tag}: roofline_pct {} not in (0, {ROOFLINE_PCT_MAX}]", pct.0));
                }
            }
            (Some(_), None) => errs.push(format!("{tag}: tg@0 present but roofline_pct missing")),
            (None, Some(_)) => errs.push(format!("{tag}: roofline_pct without tg@0")),
            (None, None) => {}
        }
        if i > 0 {
            let kind = p.decision.reason.kind;
            if p.decision.verdict == Verdict::Baseline || kind == ReasonKind::Baseline {
                errs.push(format!("{tag}: candidate carries the baseline verdict"));
            }
            if (p.decision.verdict == Verdict::Keep) != (kind == ReasonKind::Cleared) {
                errs.push(format!("{tag}: verdict {:?} with reason {kind:?}", p.decision.verdict));
            }
            if !p.series.is_empty() && p.kld.is_none() && p.failure.is_none() {
                errs.push(format!("{tag}: measured candidate without KLD and without a failure (a KLD failure must be recorded as one)"));
            }
            if p.kld.is_none() && p.failure.as_deref().is_some_and(|f| f.starts_with("kld:")) && kind != ReasonKind::Crashed {
                errs.push(format!("{tag}: KLD failure reported as {kind:?}, not crashed"));
            }
            if p.decision.verdict == Verdict::Keep && (p.kld.is_none() || p.failure.is_some()) {
                errs.push(format!("{tag}: kept without KLD or despite a failure"));
            }
        }
    }
    errs
}

/// The v1.1 quality gate, relative to the control (contract.rs "Keep rule" 4): `Err` is the
/// rejection detail.
pub fn quality_gate(t: &Thresholds, control: Quality, q: Quality) -> Result<(), String> {
    if control.mean_kld.0 > t.control_kld_cap.0 {
        return Err(format!("control KLD {:.4} > cap {:.4}", control.mean_kld.0, t.control_kld_cap.0));
    }
    if q.mean_kld.0 > control.mean_kld.0 + t.kld_margin.0 {
        return Err(format!("KLD {:.4} > control {:.4} + {:.4}", q.mean_kld.0, control.mean_kld.0, t.kld_margin.0));
    }
    if q.same_top_pct.0 < control.same_top_pct.0 - t.same_top_margin_pct.0 {
        return Err(format!(
            "same-top {:.1}% < control {:.1}% - {:.1}%",
            q.same_top_pct.0, control.same_top_pct.0, t.same_top_margin_pct.0
        ));
    }
    Ok(())
}

/// What the keep rule must say about `cand` given the measured numbers, recomputed from samples,
/// and the control's quality.
pub fn expected_kind(w: &Workload, t: &Thresholds, repeats: u32, control: Quality, base: &Point, cand: &Point) -> ReasonKind {
    let count_at = |d| cand.samples.iter().filter(|s| s.depth == d).count();
    if cand.failure.is_some() || cand.kld.is_none() || w.depths.iter().any(|d| count_at(*d) < repeats as usize) {
        return ReasonKind::Crashed;
    }
    if cand.config == base.config {
        return ReasonKind::IdenticalConfig;
    }
    let pair = |s: Series| series_stats(&base.samples, s).zip(series_stats(&cand.samples, s));
    for s in w.series() {
        match pair(s) {
            Some((b, c)) if b.n >= t.min_samples && c.n >= t.min_samples => {}
            _ => return ReasonKind::TooFewSamples,
        }
    }
    match cand.kld.map(|q| quality_gate(t, control, q)) {
        Some(Ok(())) => {}
        _ => return ReasonKind::QualityGate,
    }
    if w.guarded().iter().any(|s| pair(*s).is_some_and(|(b, c)| delta_pct(&b, &c) < -w.guard_pct.0)) {
        return ReasonKind::GuardRegression;
    }
    // Every workload series, the objective included, was present above.
    let Some((b, c)) = pair(w.objective) else {
        return ReasonKind::TooFewSamples;
    };
    let se = (b.stdev.0.powi(2) / f64::from(b.n) + c.stdev.0.powi(2) / f64::from(c.n)).sqrt();
    if c.mean.0 - b.mean.0 - t.noise_k * se <= 0.0 {
        ReasonKind::NoiseGate
    } else {
        ReasonKind::Cleared
    }
}

/// The recipe's winner: the kept point with the highest objective mean, else the baseline.
pub fn expected_winner(r: &Results, objective: Series) -> &Point {
    let mean = |p: &Point| p.series.iter().find(|x| x.series == objective).map_or(0.0, |x| x.stats.mean.0);
    r.points
        .iter()
        .filter(|p| p.decision.verdict == Verdict::Keep)
        .max_by(|x, y| mean(x).total_cmp(&mean(y)))
        .unwrap_or(&r.baseline)
}

/// Fields a draft casebook entry must state (strix-casebook CLAUDE.md: stack, numbers, verdict),
/// plus every measured series by its label (`pp@32768`), the control, and the KLD context used.
pub fn casebook_missing(text: &str, model_file: &str, series: &[Series], kld_ctx: u32) -> Vec<String> {
    let mut required: Vec<String> = ["# ", "## Stack", "## Result", "kernel", "llama.cpp", "profile", "-fa", "tok/s", "KLD", "roofline", "|", "control"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    required.push(model_file.to_string());
    required.push(format!("-c {kld_ctx} "));
    required.extend(series.iter().map(ToString::to_string));
    required.into_iter().filter(|s| !text.contains(s.as_str())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuner::contract::*;

    fn cfg() -> Config {
        Config { backend: Backend::Vulkan, fa: true, ub: 512, b: 2048, ctk: KvType::F16, ctv: KvType::F16, ctx: 8192, np: 1, hipblaslt: false }
    }

    fn pp32k() -> Series {
        Series { metric: Metric::Pp, depth: Depth(32768) }
    }

    /// `repeats` samples per depth of `(pp, tg)`, series stats derived, roofline % from tg@0.
    fn point(config: Config, rows: &[(u32, f64, f64)], verdict: Verdict, kind: ReasonKind, kld: Option<Quality>) -> Point {
        let samples: Vec<Sample> = rows
            .iter()
            .flat_map(|&(d, pp, tg)| (0..3).map(move |i| Sample { depth: Depth(d), pp_tok_s: TokS(pp + f64::from(i)), tg_tok_s: TokS(tg + 0.1 * f64::from(i)) }))
            .collect();
        let depths: BTreeSet<Depth> = samples.iter().map(|s| s.depth).collect();
        let series: Vec<SeriesStats> = [Metric::Pp, Metric::Tg]
            .into_iter()
            .flat_map(|metric| depths.iter().map(move |&depth| Series { metric, depth }))
            .filter_map(|s| series_stats(&samples, s).map(|stats| SeriesStats { series: s, stats }))
            .collect();
        let tg0 = series.iter().find(|x| x.series == Series::TG0).map(|x| x.stats.mean.0);
        Point {
            config,
            repeats: 3,
            samples,
            failure: None,
            series,
            kld,
            roofline_pct: tg0.map(|m| Pct(m / 54.26 * 100.0)),
            decision: Decision { verdict, reason: Reason { kind, detail: "x".into() } },
        }
    }

    fn good_kld() -> Option<Quality> {
        Some(Quality { mean_kld: Kld(0.037), same_top_pct: Pct(87.0) })
    }

    /// The 512-ctx reorder-only floor (casebook §2: ub 1024 f16/f16, 0.032 / 87.7%).
    fn control() -> Quality {
        Quality { mean_kld: Kld(0.032), same_top_pct: Pct(87.7) }
    }

    fn results() -> Results {
        Results {
            schema_version: SCHEMA_VERSION,
            host: Host::Strix,
            model: ModelRef { path: "m.gguf".into(), bytes: 1 },
            stack: Stack { llama_cpp: "v0.6.0 d81235049384".into(), kernel: "7.0.0-38".into(), platform_profile: "performance".into(), driver: "Mesa 26.0.8".into() },
            roofline: Roofline { hardware_id: "strix".into(), decode_bw_gbs: 195.0, active_gb_per_token: 3.594, expected_tg_tok_s: TokS(54.26) },
            started_utc: "2026-10-19T00:00:00Z".into(),
            baseline: point(cfg(), &[(0, 1015.0, 54.2), (32768, 333.3, 40.8)], Verdict::Baseline, ReasonKind::Baseline, None),
            control: Control { config: Config { ub: 1024, ..cfg() }, kld: control() },
            points: vec![point(Config { backend: Backend::Hip, hipblaslt: true, ..cfg() }, &[(0, 976.4, 48.6), (32768, 493.3, 37.9)], Verdict::Reject, ReasonKind::GuardRegression, good_kld())],
        }
    }

    #[test]
    fn control_rule_doubles_then_halves() {
        assert_eq!(control_config(cfg()).map(|c| c.ub), Some(1024));
        assert_eq!(control_config(Config { ub: 1024, ..cfg() }).map(|c| c.ub), Some(2048), "doubling to exactly b is allowed");
        assert_eq!(control_config(Config { ub: 2048, ..cfg() }).map(|c| c.ub), Some(1024), "ub == b halves");
        assert_eq!(control_config(Config { ub: 1, b: 1, ..cfg() }), None);
        let c = control_config(Config { backend: Backend::Hip, hipblaslt: true, ..cfg() }).unwrap();
        assert_eq!(Config { ub: 512, ..c }, Config { backend: Backend::Hip, hipblaslt: true, ..cfg() }, "only ub changes");
    }

    #[test]
    fn results_validation() {
        let good = results();
        assert!(validate_results(&good).is_empty(), "{:?}", validate_results(&good));

        let mut v = results();
        v.schema_version = 1;
        assert!(validate_results(&v).iter().any(|e| e.contains("schema_version")));

        // v1.1: the control must be the rule's config. A grid point may share it (A5's ub 1024): the
        // control is a separate quality measurement, not a reason to drop a candidate.
        let mut v = results();
        v.control.config.ub = 256;
        assert!(validate_results(&v).iter().any(|e| e.contains("control config")));
        let mut v = results();
        v.control.config.ctv = KvType::Q8_0;
        assert!(validate_results(&v).iter().any(|e| e.contains("control config")), "the control changes nothing but ub");
        let mut v = results();
        v.points[0].config = v.control.config;
        assert!(!validate_results(&v).iter().any(|e| e.contains("control")), "{:?}", validate_results(&v));

        // v1.1: no KLD on a measured candidate is a recorded failure and a crash, never a quality verdict.
        let mut v = results();
        v.points[0].kld = None;
        v.points[0].failure = Some("kld: exit 1: ggml_vulkan: device lost".into());
        assert!(validate_results(&v).iter().any(|e| e.contains("not crashed")), "GuardRegression with a kld: failure");
        v.points[0].decision.reason.kind = ReasonKind::Crashed;
        assert!(validate_results(&v).is_empty(), "{:?}", validate_results(&v));

        let mut v = results();
        v.points[0].roofline_pct = Some(Pct(0.0));
        assert!(validate_results(&v).iter().any(|e| e.contains("roofline_pct")));
        v.points[0].roofline_pct = None;
        assert!(validate_results(&v).iter().any(|e| e.contains("roofline_pct missing")));

        let mut v = results();
        v.points[0].series.pop();
        assert!(validate_results(&v).iter().any(|e| e.contains("every metric at every sampled depth")), "tg@32768 dropped");

        let mut v = results();
        v.points[0].series[1].stats.mean.0 += 1.0;
        assert!(validate_results(&v).iter().any(|e| e.contains("do not match the samples")));

        let mut v = results();
        v.points[0].kld = None;
        assert!(validate_results(&v).iter().any(|e| e.contains("without KLD and without a failure")));

        let mut v = results();
        v.baseline.decision.verdict = Verdict::Reject;
        assert!(validate_results(&v).iter().any(|e| e.contains("baseline verdict")));

        let mut v = results();
        v.points[0].decision.verdict = Verdict::Keep;
        assert!(validate_results(&v).iter().any(|e| e.contains("verdict Keep with reason GuardRegression")));
        v.points[0].decision.reason.kind = ReasonKind::Cleared;
        v.points[0].failure = Some("boom".into());
        assert!(validate_results(&v).iter().any(|e| e.contains("kept")));

        let mut v = results();
        let extra = v.points[0].samples[0];
        v.points[0].samples.push(extra);
        assert!(validate_results(&v).iter().any(|e| e.contains("> repeats")));
    }

    /// The casebook tradeoff through the oracle: HIP +48% pp@32768, -10% tg@0, -7% tg@32768.
    #[test]
    fn keep_rule_oracle_follows_the_contract_order() {
        let r = results();
        let t = Thresholds::default();
        let (base, hip, ctl) = (&r.baseline, &r.points[0], r.control.kld);
        let two = |objective, guard_pct| Workload { depths: vec![Depth(0), Depth(32768)], objective, guard_pct: Pct(guard_pct), ..Workload::default() };
        assert_eq!(expected_kind(&two(pp32k(), 3.0), &t, 3, ctl, base, hip), ReasonKind::GuardRegression);
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, hip), ReasonKind::Cleared);
        assert_eq!(expected_kind(&two(Series::TG0, 3.0), &t, 3, ctl, base, hip), ReasonKind::GuardRegression, "tg@32768 and pp@0 regress");
        assert_eq!(expected_kind(&two(Series::TG0, 15.0), &t, 3, ctl, base, hip), ReasonKind::NoiseGate, "the objective's own loss is the noise gate's");

        let bad_kld = Point { kld: Some(Quality { mean_kld: Kld(0.07), same_top_pct: Pct(83.0) }), ..hip.clone() };
        assert_eq!(expected_kind(&two(pp32k(), 3.0), &t, 3, ctl, base, &bad_kld), ReasonKind::QualityGate, "quality before guard");
        let crashed = Point { failure: Some("exit 139".into()), ..hip.clone() };
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, &crashed), ReasonKind::Crashed);
        let no_kld = Point { kld: None, failure: Some("kld: timed out: ...".into()), ..hip.clone() };
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, &no_kld), ReasonKind::Crashed, "a KLD failure is a crash");
        let silent = Point { kld: None, ..hip.clone() };
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, &silent), ReasonKind::Crashed, "no reading at all is a crash, not a quality verdict");
        let mut short = hip.clone();
        short.samples.retain(|s| s.depth == Depth(0));
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, &short), ReasonKind::Crashed, "a missing depth is a crash");
        let same = Point { config: base.config, ..hip.clone() };
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 3, ctl, base, &same), ReasonKind::IdenticalConfig);
        let mut thin = hip.clone();
        thin.samples.truncate(4);
        assert_eq!(expected_kind(&two(pp32k(), 15.0), &t, 1, ctl, base, &thin), ReasonKind::TooFewSamples);
    }

    /// v1.1: the gate is relative to the control, with the casebook's 32k numbers (§3): control
    /// 0.079 / 82.4%, q8_0/q8_0 0.0963 / 79.9% passes, 0.11 fails; the W4A4 regression fails any
    /// honest control; a control over the cap rejects everything with its own detail.
    #[test]
    fn quality_gate_is_relative_to_the_control() {
        let t = Thresholds::default();
        let q = |k, s| Quality { mean_kld: Kld(k), same_top_pct: Pct(s) };
        let deep = q(0.079, 82.4);
        assert_eq!(quality_gate(&t, deep, q(0.0963, 79.9)), Ok(()), "q8_0/q8_0 at 32k: +0.017 / -2.5");
        assert_eq!(quality_gate(&t, deep, q(0.0859, 81.4)), Ok(()), "f16/q8_0 at 32k");
        assert_eq!(quality_gate(&t, deep, q(0.11, 82.0)), Err("KLD 0.1100 > control 0.0790 + 0.0200".into()));
        assert_eq!(quality_gate(&t, deep, q(0.0991, 82.0)), Err("KLD 0.0991 > control 0.0790 + 0.0200".into()), "just past");
        let shallow = q(0.035, 87.0);
        assert_eq!(quality_gate(&t, shallow, q(0.1083, 74.6)).unwrap_err(), "KLD 0.1083 > control 0.0350 + 0.0200", "W4A4");
        assert_eq!(quality_gate(&t, shallow, q(0.035, 84.0)), Ok(()), "same-top exactly at control - margin passes");
        assert_eq!(quality_gate(&t, shallow, q(0.035, 83.9)), Err("same-top 83.9% < control 87.0% - 3.0%".into()));
        assert_eq!(quality_gate(&t, q(0.25, 60.0), q(0.03, 90.0)), Err("control KLD 0.2500 > cap 0.2000".into()), "a broken control judges nothing");
        assert_eq!(quality_gate(&t, q(0.2, 60.0), q(0.03, 90.0)), Ok(()), "exactly at the cap is still a control");
    }

    /// Fixtures o/p: objective pp@0 +10%, guarded tg@0 at exactly -3.0% (50 -> 48.5) then -3.1%.
    #[test]
    fn guard_bound_is_inclusive() {
        let base = point(cfg(), &[(0, 1000.0, 50.0)], Verdict::Baseline, ReasonKind::Baseline, None);
        let mut cand = point(Config { ub: 1024, ..cfg() }, &[(0, 1100.0, 50.0)], Verdict::Keep, ReasonKind::Cleared, good_kld());
        let set_tg = |p: &mut Point, tg: f64| p.samples.iter_mut().for_each(|s| s.tg_tok_s = TokS(tg));
        let mut base = base;
        set_tg(&mut base, 50.0);
        set_tg(&mut cand, 48.5);
        let w = Workload { objective: Series { metric: Metric::Pp, depth: Depth(0) }, ..Workload::default() };
        let d = delta_pct(&series_stats(&base.samples, Series::TG0).unwrap(), &series_stats(&cand.samples, Series::TG0).unwrap());
        assert_eq!(d, -3.0, "the bound must be hit exactly for the fixture to mean anything");
        assert_eq!(expected_kind(&w, &Thresholds::default(), 3, control(), &base, &cand), ReasonKind::Cleared);
        set_tg(&mut cand, 48.45);
        assert_eq!(expected_kind(&w, &Thresholds::default(), 3, control(), &base, &cand), ReasonKind::GuardRegression);
    }

    #[test]
    fn winner_is_the_best_kept_by_objective_else_baseline() {
        let mut r = results();
        assert!(std::ptr::eq(expected_winner(&r, pp32k()), &r.baseline), "nothing kept");
        r.points[0].decision = Decision { verdict: Verdict::Keep, reason: Reason { kind: ReasonKind::Cleared, detail: "x".into() } };
        assert_eq!(expected_winner(&r, pp32k()).config.backend, Backend::Hip);
        let slower_pp = point(Config { ub: 1024, ..cfg() }, &[(0, 1100.0, 60.0), (32768, 400.0, 45.0)], Verdict::Keep, ReasonKind::Cleared, good_kld());
        r.points.push(slower_pp);
        assert_eq!(expected_winner(&r, pp32k()).config.backend, Backend::Hip, "493 > 400 at pp@32768");
        assert_eq!(expected_winner(&r, Series::TG0).config.ub, 1024, "60 > 48.6 at tg@0");
    }

    #[test]
    fn casebook_entry_fields() {
        let entry = "# tuner sweep: gpt-oss-120b on strix, 2026-10-19\n\n## Stack\nkernel 7.0.0-38, profile performance, Mesa RADV, llama.cpp v0.6.0 d81235049384 build-vulkan, -fa on\n\
                     KLD `-c 512 --chunks 32`; control (-ub 1024): KLD 0.0320, same-top 87.7%\n\n## Result\n| config | pp@0 tok/s | tg@0 tok/s | KLD | % roofline |\n|---|---|---|---|---|\n| baseline | 1015.0 | 54.1 | - | 99.7 |\n\nRaw: results.json\n";
        let v0 = [Series { metric: Metric::Pp, depth: Depth(0) }, Series::TG0];
        assert!(casebook_missing(entry, "gpt-oss-120b", &v0, 512).is_empty());
        let missing = casebook_missing(&entry.replace("## Stack", "## Setup"), "gpt-oss-120b", &v0, 512);
        assert_eq!(missing, vec!["## Stack".to_string()]);
        assert!(casebook_missing(entry, "qwen", &v0, 512).contains(&"qwen".to_string()));
        assert_eq!(casebook_missing(entry, "gpt-oss-120b", &[pp32k()], 512), vec!["pp@32768".to_string()]);
        assert_eq!(casebook_missing(entry, "gpt-oss-120b", &v0, 32768), vec!["-c 32768 ".to_string()], "the entry must state the KLD context");
        assert_eq!(casebook_missing(&entry.replace("control", "ctrl"), "gpt-oss-120b", &v0, 512), vec!["control".to_string()]);
    }
}
