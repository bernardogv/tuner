//! What a sweep measures: throughput series keyed by (metric, depth), and which one must improve.
//! Pure types; part of `contract`.

use serde::{Deserialize, Serialize};
use std::fmt;

use super::{Pct, Stats};

/// Which llama-bench throughput a series measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    /// Prompt processing (prefill): `-p 512`.
    Pp,
    /// Token generation (decode): `-n 128`.
    Tg,
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Metric::Pp => "pp",
            Metric::Tg => "tg",
        })
    }
}

/// Tokens already in the KV cache when a test runs: llama-bench `-d`. v0 only ever measured 0.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Depth(pub u32);

/// One throughput series. `Ord` is (metric, depth): pp@0, pp@32768, tg@0, tg@32768.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Series {
    pub metric: Metric,
    pub depth: Depth,
}

impl Series {
    /// v0's only judged series, and the one the roofline % is computed from.
    pub const TG0: Series = Series {
        metric: Metric::Tg,
        depth: Depth(0),
    };

    fn tg0() -> Series {
        Series::TG0
    }
}

/// `pp@32768`: the spelling reason details, casebook entries and the harness use for a series.
impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.metric, self.depth.0)
    }
}

/// What a sweep measures and what must improve: `[workload]` in sweep.toml and `workload` in a
/// decide fixture. Every field has a default, and the defaults are v0: depth 0 only, tg@0 must
/// improve, nothing else may regress more than 3%. Omit the table and a v0 spec means what it meant.
///
/// - `guard_pct` 3.0: the strix casebook treats sub-3% deltas on one-run comparisons as noise
///   (2026-10-05-gpt-oss-120b-vulkan-vs-hip), so a regression inside that band is not a tradeoff.
///   The HIP-at-32k tradeoff (+48% pp@32768 for -10% tg@0) is exactly what this exposes: an
///   operator who wants it says so with `guard_pct` large enough to accept the decode loss.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workload {
    /// llama-bench `-d`; every point is measured at every depth listed, in this order.
    #[serde(default = "default_depths")]
    pub depths: Vec<Depth>,
    /// The series the noise gate judges. Its depth must be in `depths`.
    #[serde(default = "Series::tg0")]
    pub objective: Series,
    /// Every series other than the objective may regress at most this much, as mean delta %
    /// (`(mean_c - mean_b) / mean_b * 100`); exactly `-guard_pct` passes.
    #[serde(default = "default_guard_pct")]
    pub guard_pct: Pct,
    /// llama-perplexity `-c` for every KLD run (baseline logits, control, candidates); the chunk
    /// count follows from it (contract docs, "Quality"). 512 is the v0 protocol; 32768 measures
    /// quality at depth and needs ~13 GB of disk on the host for gpt-oss.
    #[serde(default = "default_kld_ctx")]
    pub kld_ctx: u32,
}

fn default_depths() -> Vec<Depth> {
    vec![Depth(0)]
}

fn default_guard_pct() -> Pct {
    Pct(3.0)
}

fn default_kld_ctx() -> u32 {
    512
}

impl Default for Workload {
    fn default() -> Self {
        Workload {
            depths: default_depths(),
            objective: Series::TG0,
            guard_pct: default_guard_pct(),
            kld_ctx: default_kld_ctx(),
        }
    }
}

impl Workload {
    /// Every series a sweep of this workload produces: both metrics at every depth, in `Series`
    /// order (so pp@0, pp@32768, tg@0, tg@32768 for depths [0, 32768]).
    pub fn series(&self) -> Vec<Series> {
        let mut out: Vec<Series> = [Metric::Pp, Metric::Tg]
            .into_iter()
            .flat_map(|metric| self.depths.iter().map(move |&depth| Series { metric, depth }))
            .collect();
        out.sort();
        out.dedup();
        out
    }

    /// Guarded series: every series except the objective, in `Series` order.
    pub fn guarded(&self) -> Vec<Series> {
        self.series().into_iter().filter(|s| *s != self.objective).collect()
    }
}

/// Mean / stdev / n of one series of a `Point`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesStats {
    pub series: Series,
    pub stats: Stats,
}

/// Candidate vs baseline on one series: `(mean_c - mean_b) / mean_b * 100`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeriesDelta {
    pub series: Series,
    pub delta_pct: Pct,
}
