//! `tuner decide --fixture` input and output: the keep rule as a black box. Pure types; part of
//! `contract`.

use serde::{Deserialize, Serialize};

use super::{Decision, Quality, ReasonKind, Sample, SeriesDelta, Thresholds, Verdict, Workload};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureBaseline {
    pub samples: Vec<Sample>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureCandidate {
    pub samples: Vec<Sample>,
    pub kld: Option<Quality>,
    pub failure: Option<String>,
    pub identical_to_baseline: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expectation {
    pub verdict: Verdict,
    pub reason_kind: ReasonKind,
    /// When set, `reason.detail` must contain this substring (e.g. a guard violator).
    #[serde(default)]
    pub detail_contains: Option<String>,
}

/// Input of `tuner decide --fixture`. `expect` is for the harness; the tuner ignores it.
/// `workload` omitted means `Workload::default()`. `control` is required: a v1 fixture (schema 2,
/// absolute thresholds, no control) does not parse and `decide` exits `FAILED` naming the field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecideFixture {
    pub schema_version: u32,
    pub notes: Option<String>,
    pub thresholds: Thresholds,
    #[serde(default)]
    pub workload: Workload,
    pub repeats: u32,
    pub baseline: FixtureBaseline,
    /// The reorder-only control's quality (`Control::kld`); its config plays no part in the rule.
    pub control: Quality,
    pub candidate: FixtureCandidate,
    pub expect: Expectation,
}

/// Output of `tuner decide`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecideOutput {
    pub schema_version: u32,
    #[serde(flatten)]
    pub decision: Decision,
    /// One per series of `Workload::series` that has samples on both sides, in `Series` order.
    pub deltas: Vec<SeriesDelta>,
}
