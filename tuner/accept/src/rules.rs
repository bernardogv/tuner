//! The harness's own assertion logic, pure and unit-tested: what a tuner output must look like.
//! Nothing here measures anything; it only checks the tuner's claims. `plan` holds the grid oracle,
//! `results` the results.json rules and the keep-rule oracle a live sweep is recomputed against.

pub mod plan;
pub mod results;

pub use plan::check_plan;
pub use results::{casebook_missing, expected_kind, expected_winner, series_stats, validate_results};

use std::collections::BTreeSet;

use tuner::contract::{Decision, Expectation, TokS};

pub const GIB: u64 = 1024 * 1024 * 1024;
/// Headroom the live check demands after the model (strix-casebook convention).
pub const REQUIRED_HEADROOM: u64 = 20 * GIB;
/// strix-casebook hardware.md / models.md: gpt-oss-120b MXFP4, Vulkan, tg128 at depth 0,
/// 54.27 (2026-10-05 anchor; 53.87 and 54.42 on later rows).
pub const STRIX_GPT_OSS_ANCHOR: TokS = TokS(54.27);
pub const ANCHOR_TOLERANCE_PCT: f64 = 10.0;

/// May the live check load this model right now? Returns the headroom in bytes.
pub fn live_gate(mem_available: u64, model_bytes: u64, running: &[String], procs: &str) -> Result<u64, String> {
    if !running.is_empty() {
        return Err(format!("llama-swap has a model resident: {running:?}"));
    }
    if !procs.trim().is_empty() {
        return Err(format!("llama.cpp process(es) alive: {}", procs.trim().replace('\n', "; ")));
    }
    match mem_available.checked_sub(model_bytes) {
        Some(h) if h >= REQUIRED_HEADROOM => Ok(h),
        Some(h) => Err(format!(
            "only {:.1} GiB would remain after a {:.1} GiB model (need >= {} GiB)",
            h as f64 / GIB as f64,
            model_bytes as f64 / GIB as f64,
            REQUIRED_HEADROOM / GIB
        )),
        None => Err(format!(
            "model ({:.1} GiB) exceeds MemAvailable ({:.1} GiB)",
            model_bytes as f64 / GIB as f64,
            mem_available as f64 / GIB as f64
        )),
    }
}

/// `pgrep` lines present after a run that were not present before.
pub fn orphans(before: &str, after: &str) -> Vec<String> {
    let seen: BTreeSet<&str> = before.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    after
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !seen.contains(l))
        .map(str::to_string)
        .collect()
}

pub fn compare_decision(expect: &Expectation, got: &Decision) -> Result<(), String> {
    if got.verdict != expect.verdict || got.reason.kind != expect.reason_kind {
        return Err(format!(
            "expected {:?}/{:?}, got {:?}/{:?} ({})",
            expect.verdict, expect.reason_kind, got.verdict, got.reason.kind, got.reason.detail
        ));
    }
    if got.reason.detail.trim().is_empty() {
        return Err("reason.detail is empty".into());
    }
    if let Some(needle) = &expect.detail_contains {
        if !got.reason.detail.contains(needle.as_str()) {
            return Err(format!("reason.detail lacks {needle:?}: {}", got.reason.detail));
        }
    }
    Ok(())
}

pub fn anchor_within(measured: TokS, anchor: TokS, tol_pct: f64) -> Result<(), String> {
    let off = (measured.0 - anchor.0).abs() / anchor.0 * 100.0;
    if off <= tol_pct {
        Ok(())
    } else {
        Err(format!("tg {:.2} is {off:.1}% from the anchor {:.2} (allowed {tol_pct}%)", measured.0, anchor.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuner::contract::*;

    #[test]
    fn headroom_gate() {
        let idle: Vec<String> = vec![];
        assert_eq!(live_gate(120 * GIB, 63 * GIB, &idle, ""), Ok(57 * GIB));
        assert!(live_gate(120 * GIB, 101 * GIB, &idle, "").is_err(), "19 GiB left");
        assert!(live_gate(120 * GIB, 100 * GIB, &idle, "").is_ok(), "exactly 20 GiB left");
        assert!(live_gate(10 * GIB, 63 * GIB, &idle, "").is_err(), "no underflow");
        assert!(live_gate(120 * GIB, GIB, &["gpt-oss".to_string()], "").is_err(), "resident in llama-swap");
        assert!(live_gate(120 * GIB, GIB, &idle, "123 llama-server -m x").is_err(), "stray process");
    }

    #[test]
    fn orphan_detection_is_set_difference() {
        assert!(orphans("", "").is_empty());
        assert!(orphans("1 llama-server a\n", "1 llama-server a\n").is_empty());
        assert_eq!(orphans("", "7 llama-bench -m x\n"), vec!["7 llama-bench -m x".to_string()]);
        assert!(orphans("7 llama-bench -m x\n", "").is_empty(), "a process that went away is not an orphan");
    }

    #[test]
    fn decision_comparison() {
        let exp = Expectation { verdict: Verdict::Reject, reason_kind: ReasonKind::QualityGate, detail_contains: None };
        let ok = Decision {
            verdict: Verdict::Reject,
            reason: Reason { kind: ReasonKind::QualityGate, detail: "KLD 0.07 > 0.05".into() },
        };
        assert!(compare_decision(&exp, &ok).is_ok());
        let wrong_kind = Decision { reason: Reason { kind: ReasonKind::NoiseGate, ..ok.reason.clone() }, ..ok.clone() };
        assert!(compare_decision(&exp, &wrong_kind).is_err());
        let wrong_verdict = Decision { verdict: Verdict::Keep, ..ok.clone() };
        assert!(compare_decision(&exp, &wrong_verdict).is_err());
        let no_detail = Decision { reason: Reason { detail: " ".into(), ..ok.reason.clone() }, ..ok.clone() };
        assert!(compare_decision(&exp, &no_detail).is_err());
        let needle = Expectation { detail_contains: Some("0.07 > 0.05".into()), ..exp.clone() };
        assert!(compare_decision(&needle, &ok).is_ok());
        let miss = Expectation { detail_contains: Some("tg@0 -10.3%".into()), ..exp };
        assert!(compare_decision(&miss, &ok).unwrap_err().contains("lacks"));
    }

    #[test]
    fn anchor_tolerance() {
        assert!(anchor_within(TokS(49.0), STRIX_GPT_OSS_ANCHOR, 10.0).is_ok());
        assert!(anchor_within(TokS(48.0), STRIX_GPT_OSS_ANCHOR, 10.0).is_err());
        assert!(anchor_within(TokS(59.6), STRIX_GPT_OSS_ANCHOR, 10.0).is_ok());
        assert!(anchor_within(TokS(60.0), STRIX_GPT_OSS_ANCHOR, 10.0).is_err());
    }

    /// Back-compat: a v0 spec / fixture without the new fields reads as depth 0, tg@0, guard 3,
    /// KLD at 512 (v1.1: `kld_ctx`).
    #[test]
    fn workload_defaults_are_v0() {
        let w = Workload::default();
        assert_eq!(w.depths, vec![Depth(0)]);
        assert_eq!(w.objective, Series::TG0);
        assert_eq!(w.guard_pct, Pct(3.0));
        assert_eq!(w.kld_ctx, 512);
        assert_eq!(w.series(), vec![Series { metric: Metric::Pp, depth: Depth(0) }, Series::TG0]);
        assert_eq!(w.guarded(), vec![Series { metric: Metric::Pp, depth: Depth(0) }]);

        let v0_spec = "schema_version = 3\nrepeats = 3\n[baseline]\nbackend = \"vulkan\"\nfa = true\nub = 512\nb = 2048\n\
                       ctk = \"f16\"\nctv = \"f16\"\nctx = 8192\nnp = 1\nhipblaslt = false\n";
        let spec: Spec = toml::from_str(v0_spec).unwrap();
        assert_eq!(spec.workload, Workload::default());
        let v1 = format!("{v0_spec}[workload]\ndepths = [0, 32768]\nobjective = {{ metric = \"pp\", depth = 32768 }}\n");
        let spec: Spec = toml::from_str(&v1).unwrap();
        assert_eq!(spec.workload.depths, vec![Depth(0), Depth(32768)]);
        assert_eq!(spec.workload.objective.to_string(), "pp@32768");
        assert_eq!(spec.workload.guard_pct, Pct(3.0), "guard keeps its default when only depths/objective are set");
        assert_eq!(spec.workload.kld_ctx, 512);
        let deep: Spec = toml::from_str(&format!("{v0_spec}[workload]\nkld_ctx = 32768\n")).unwrap();
        assert_eq!(deep.workload.kld_ctx, 32768);

        let v0_sample: Sample = serde_json::from_str(r#"{"pp_tok_s": 1000.0, "tg_tok_s": 54.0}"#).unwrap();
        assert_eq!(v0_sample.depth, Depth(0));
    }

    /// v1.1 thresholds are margins over the control plus the control cap; a v1 fixture's
    /// absolute `kld_max` no longer parses, and `control` is required.
    #[test]
    fn thresholds_are_relative_and_fixtures_carry_a_control() {
        let t = Thresholds::default();
        assert_eq!((t.kld_margin, t.same_top_margin_pct, t.control_kld_cap, t.noise_k, t.min_samples), (Kld(0.02), Pct(3.0), Kld(0.2), 2.0, 3));
        assert!(serde_json::from_str::<Thresholds>(r#"{"kld_max":0.05,"same_top_min_pct":85.0,"noise_k":2.0,"min_samples":3}"#).is_err());
        let fixture = r#"{"schema_version":3,"notes":null,"thresholds":{"kld_margin":0.02,"same_top_margin_pct":3.0,"control_kld_cap":0.2,"noise_k":2.0,"min_samples":3},
            "repeats":3,"baseline":{"samples":[]},"candidate":{"samples":[],"kld":null,"failure":null,"identical_to_baseline":false},
            "expect":{"verdict":"reject","reason_kind":"crashed"}}"#;
        let err = serde_json::from_str::<DecideFixture>(fixture).unwrap_err().to_string();
        assert!(err.contains("control"), "{err}");
        let with_control = fixture.replace(r#""repeats":3,"#, r#""repeats":3,"control":{"mean_kld":0.035,"same_top_pct":87.0},"#);
        assert_eq!(serde_json::from_str::<DecideFixture>(&with_control).unwrap().control, Quality { mean_kld: Kld(0.035), same_top_pct: Pct(87.0) });
    }
}
