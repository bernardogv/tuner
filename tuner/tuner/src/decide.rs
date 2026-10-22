//! The keep rule (contract.rs "Keep rule"): pure, so `tuner decide` and `sweep` share one copy.

use tuner::contract::{
    Decision, Depth, Metric, Pct, Quality, Reason, ReasonKind, Sample, Series, SeriesDelta, Stats, Thresholds, TokS, Verdict,
    Workload,
};

/// What a candidate brings to the rule besides its samples.
pub struct Candidate<'a> {
    pub samples: &'a [Sample],
    pub kld: Option<Quality>,
    pub failure: Option<&'a str>,
    pub identical_to_baseline: bool,
}

/// Mean and sample stdev (n-1) of one series; `None` when it has no samples.
pub fn stats(samples: &[Sample], s: Series) -> Option<Stats> {
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
    let var = if n > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    Some(Stats {
        mean: TokS(mean),
        stdev: TokS(var.sqrt()),
        n: u32::try_from(n).unwrap_or(u32::MAX),
    })
}

/// Candidate vs baseline on every workload series present on both sides, in `Series` order.
pub fn deltas(w: &Workload, baseline: &[Sample], candidate: &[Sample]) -> Vec<SeriesDelta> {
    w.series()
        .into_iter()
        .filter_map(|s| {
            let (b, c) = (stats(baseline, s)?, stats(candidate, s)?);
            Some(SeriesDelta {
                series: s,
                delta_pct: Pct((c.mean.0 - b.mean.0) / b.mean.0 * 100.0),
            })
        })
        .collect()
}

fn reject(kind: ReasonKind, detail: String) -> Decision {
    Decision {
        verdict: Verdict::Reject,
        reason: Reason { kind, detail },
    }
}

/// `control` is the reorder-only control's quality (contract.rs "Quality"); the quality gate is
/// relative to it, because any kernel change drifts KLD (0.03 at 512 ctx, 0.08 at 32k).
pub fn decide(t: &Thresholds, w: &Workload, repeats: u32, baseline: &[Sample], control: Quality, cand: &Candidate) -> Decision {
    let at = |d: Depth| u32::try_from(cand.samples.iter().filter(|x| x.depth == d).count()).unwrap_or(u32::MAX);
    if let Some(f) = cand.failure {
        let n = w.depths.iter().map(|&d| at(d)).min().unwrap_or(0);
        return reject(ReasonKind::Crashed, format!("failed after {n}/{repeats} repeats: {f}"));
    }
    if let Some(&d) = w.depths.iter().find(|&&d| at(d) < repeats) {
        return reject(
            ReasonKind::Crashed,
            format!("only {}/{repeats} repeats produced samples at depth {}", at(d), d.0),
        );
    }
    let Some(q) = cand.kld else {
        return reject(ReasonKind::Crashed, "no KLD reading".into());
    };
    if cand.identical_to_baseline {
        return reject(ReasonKind::IdenticalConfig, "candidate config equals the baseline".into());
    }
    for s in w.series() {
        let n = |xs: &[Sample]| stats(xs, s).map_or(0, |st| st.n);
        let (n_b, n_c) = (n(baseline), n(cand.samples));
        if n_b < t.min_samples || n_c < t.min_samples {
            return reject(
                ReasonKind::TooFewSamples,
                format!("{s}: n baseline {n_b} / candidate {n_c} < {}", t.min_samples),
            );
        }
    }
    if control.mean_kld.0 > t.control_kld_cap.0 {
        return reject(
            ReasonKind::QualityGate,
            format!("control KLD {:.4} > cap {:.4}", control.mean_kld.0, t.control_kld_cap.0),
        );
    }
    if q.mean_kld.0 > control.mean_kld.0 + t.kld_margin.0 {
        return reject(
            ReasonKind::QualityGate,
            format!("KLD {:.4} > control {:.4} + {:.4}", q.mean_kld.0, control.mean_kld.0, t.kld_margin.0),
        );
    }
    if q.same_top_pct.0 < control.same_top_pct.0 - t.same_top_margin_pct.0 {
        return reject(
            ReasonKind::QualityGate,
            format!(
                "same-top {:.1}% < control {:.1}% - {:.1}%",
                q.same_top_pct.0, control.same_top_pct.0, t.same_top_margin_pct.0
            ),
        );
    }
    let violators: Vec<String> = deltas(w, baseline, cand.samples)
        .into_iter()
        .filter(|d| d.series != w.objective && d.delta_pct.0 < -w.guard_pct.0)
        .map(|d| format!("{} {:+.1}% < -{:.1}%", d.series, d.delta_pct.0, w.guard_pct.0))
        .collect();
    if !violators.is_empty() {
        return reject(ReasonKind::GuardRegression, violators.join("; "));
    }
    // TooFewSamples above guarantees both sides have the objective series.
    let (Some(b), Some(c)) = (stats(baseline, w.objective), stats(cand.samples, w.objective)) else {
        return reject(ReasonKind::TooFewSamples, format!("{}: a side has no samples", w.objective));
    };
    let delta = c.mean.0 - b.mean.0;
    let se = (b.stdev.0.powi(2) / f64::from(b.n) + c.stdev.0.powi(2) / f64::from(c.n)).sqrt();
    let lower = delta - t.noise_k * se;
    let numbers = format!(
        "{} {:.2} -> {:.2} tok/s, delta {delta:+.2}, lower bound {lower:+.2} (k {} x SE {se:.2}); KLD {:.3}",
        w.objective, b.mean.0, c.mean.0, t.noise_k, q.mean_kld.0
    );
    if lower <= 0.0 {
        return reject(ReasonKind::NoiseGate, numbers);
    }
    Decision {
        verdict: Verdict::Keep,
        reason: Reason {
            kind: ReasonKind::Cleared,
            detail: numbers,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuner::contract::Kld;

    fn s(tg: f64) -> Sample {
        at(0, 1000.0, tg)
    }

    fn at(depth: u32, pp: f64, tg: f64) -> Sample {
        Sample {
            depth: Depth(depth),
            pp_tok_s: TokS(pp),
            tg_tok_s: TokS(tg),
        }
    }

    fn good_kld() -> Option<Quality> {
        Some(Quality {
            mean_kld: Kld(0.035),
            same_top_pct: Pct(87.0),
        })
    }

    /// The 512-ctx reorder-only floor: the control every v0-shaped fixture is judged against.
    fn control() -> Quality {
        Quality {
            mean_kld: Kld(0.035),
            same_top_pct: Pct(87.0),
        }
    }

    #[test]
    fn stdev_is_sample_not_population() {
        let st = stats(&[s(1.0), s(3.0)], Series::TG0).unwrap();
        assert_eq!(st.mean.0, 2.0);
        assert!((st.stdev.0 - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn stats_and_deltas_are_per_series() {
        let base = [at(0, 1000.0, 50.0), at(32768, 300.0, 40.0)];
        let cand = [at(0, 1000.0, 45.0), at(32768, 450.0, 40.0)];
        let pp32k = Series { metric: Metric::Pp, depth: Depth(32768) };
        assert_eq!(stats(&base, pp32k).unwrap().mean.0, 300.0);
        assert_eq!(stats(&base, Series { metric: Metric::Pp, depth: Depth(1) }), None);
        let w = Workload { depths: vec![Depth(0), Depth(32768)], ..Workload::default() };
        let d: Vec<(String, f64)> = deltas(&w, &base, &cand).iter().map(|x| (x.series.to_string(), x.delta_pct.0)).collect();
        assert_eq!(d, vec![("pp@0".into(), 0.0), ("pp@32768".into(), 50.0), ("tg@0".into(), -10.0), ("tg@32768".into(), 0.0)]);
    }

    #[test]
    fn crash_wins_over_a_faster_result() {
        let base = [s(50.0), s(50.0), s(50.0)];
        let c = Candidate {
            samples: &[s(90.0)],
            kld: good_kld(),
            failure: None,
            identical_to_baseline: false,
        };
        assert_eq!(decide(&Thresholds::default(), &Workload::default(), 3, &base, control(), &c).reason.kind, ReasonKind::Crashed);
    }

    #[test]
    fn lower_bound_exactly_zero_is_rejected() {
        // No variance and no gain: the lower bound is exactly 0, which is not a gain.
        let base = [s(50.0), s(50.0), s(50.0)];
        let c = Candidate {
            samples: &base,
            kld: good_kld(),
            failure: None,
            identical_to_baseline: false,
        };
        assert_eq!(decide(&Thresholds::default(), &Workload::default(), 3, &base, control(), &c).reason.kind, ReasonKind::NoiseGate);
    }
}
