//! The tuner's CLI/JSON contract: pure types, no logic. The `accept` harness parses every
//! tuner output with these, so a field here is a field the harness checks.
//!
//! # CLI
//! ```text
//! tuner plan   --host <h> --model <gguf> --spec <sweep.toml> [--recipes <dir>]
//!     -> `Plan` JSON on stdout. No side effects, no ssh.
//! tuner sweep  --host <h> --model <gguf> --spec <sweep.toml> [--recipes <dir>] --out <dir>
//!              [--reserve-gib N] [--spark-mode-ack <mode>]
//!     -> writes <out>/results.json (`Results`), <out>/casebook-entry.md, <out>/kld-*.log (every
//!        llama-perplexity run, see "Quality"), and <recipes>/<host>/<model stem>.json (`Recipe`,
//!        the winner; the baseline if nothing was kept).
//! tuner decide --fixture <file.json>
//!     -> reads a `DecideFixture`, prints a `DecideOutput`. Pure; the keep rule, black-box.
//! ```
//!
//! # Preflight (sweep), exit `exit::REFUSED`, nothing started on the host
//! - `MemAvailable - model bytes < reserve` (default `DEFAULT_RESERVE_GIB`).
//! - llama-swap `:8089/running` lists a model, or any llama-server/llama-bench/llama-perplexity
//!   process exists. Never load beside a resident model.
//! - Free disk under `$HOME` on the host (where the run dir lives) is below the baseline logits
//!   file's size, `bench::kld_base_bytes(n_vocab, kld_ctx)` = `n_vocab * 2 * scored tokens`
//!   (gpt-oss, 201088 tokens: 3.3 GB at `kld_ctx` 512, 13.2 GB at 32768). `n_vocab` is the length
//!   of `tokenizer.ggml.tokens` in the GGUF header the sweep already fetches for the roofline
//!   (`gguf::Active::n_vocab`), so the header is read before the refusal; a header without it uses
//!   `FALLBACK_N_VOCAB`, the largest vocab in common use, so the estimate errs high.
//! - Spark hosts (`Host::is_spark`): refuse unless `--spark-mode-ack <mode>` is given. The flag is
//!   the operator's statement that they ran `spark-mode switch <mode>` and `<mode>` serves nothing.
//!   With the flag, the tuner runs `~/bin/spark-mode status` (read-only) and still refuses unless the
//!   reported `mode:` equals `<mode>`, every listed unit is `inactive`, and the headroom gate passes.
//!   The tuner never runs `spark-mode switch`, never touches a systemd unit, and in v0 never
//!   benchmarks on a Spark at all (the ack path ends in `exit::UNIMPLEMENTED`).
//!
//! # Processes
//! Every remote process the tuner starts runs in its own session (`setsid`) with its pgid recorded
//! under the run dir, and is reaped (TERM, then KILL) on normal exit, error, and ctrl-c.
//!
//! # Grid expansion (`plan`)
//! `points[0]` is the baseline. Then the warm-start config, if a recipe exists for
//! `<recipes>/<host>/<model stem>.json`. Then the cartesian product of every axis listed under
//! `[grid]` (unlisted axes keep the baseline's value), normalized and deduplicated:
//! `hipblaslt = true` with `backend = "vulkan"` becomes `false`; the baseline and the warm-start
//! config are removed from the product. The product is sorted by `Config`'s derived `Ord`
//! (field declaration order), so two `plan` runs are byte-identical.
//!
//! # Workload (v1): depths, objective, guard
//! A spec's `[workload]` (`Workload`) says what is measured and what must improve. llama-bench runs
//! `-p 512 -n 128 -d <depths comma list>` (always passed, `-d 0` for the default), one row per
//! (metric, depth) with `samples_ts` per repeat; the tuner pairs the pp and tg rows of one depth
//! by repeat index into `Sample { depth, pp_tok_s, tg_tok_s }`, depth-major (every repeat at
//! `depths[0]`, then `depths[1]`, ...). A `Point` reports a `SeriesStats` for every series that has
//! samples, and `roofline_pct` only for tg@0 (the roofline is a decode number).
//!
//! # Quality (v1.1): control, relative gate, KLD at depth, KLD logs
//! Every KLD number is llama-perplexity on wikitext-2 test, `-c <kld_ctx> --chunks <n>` with
//! `n = bench::kld_chunks(kld_ctx) = max(2, 16384 / kld_ctx)` (`Workload::kld_ctx`, default 512:
//! `-c 512 --chunks 32`, the v0 protocol; 32768: `-c 32768 --chunks 2`, the casebook's 32k
//! protocol). llama-perplexity scores the second half of each chunk, so 8k tokens are scored at 512
//! and 32k at 32768; the floor of 2 chunks keeps the `±` on `Mean KLD` defined. The baseline's run
//! saves its logits to `<run_dir>/base.kld` (size under "Preflight"); the file is deleted on reap.
//! `kld_ctx` is passed through as given; a value llama-perplexity cannot run with fails the
//! baseline logits step below.
//!
//! The same config is bit-deterministic (KLD 0) but ANY kernel change drifts, and the drift grows
//! with context: a reorder-only `-ub` change measured 0.032 / 87.7% at 512 and 0.079 / 82.4% at 32k
//! (strix-casebook 2026-10-21-tuner-v1-depth-sweep), so an absolute gate rejects harmless changes
//! at depth. Every sweep therefore measures a **control**: the baseline config with only `ub`
//! changed, `bench::control_config`: `ub*2` if `ub*2 <= b`, else `ub/2` (512/2048 -> 1024,
//! 1024/2048 -> 2048, 2048/2048 -> 1024). Doubling first because the margins below were calibrated
//! against the doubled control; the `<= b` bound keeps `ub <= b`, which llama.cpp would otherwise
//! clamp back to the baseline. The one config with no control (`ub` 1, `b` 1) is refused by
//! `read_spec`. The control is KLD'd against the baseline logits right after them and before any
//! candidate; it is not benched (no rule consumes its throughput, and a 32k bench costs ~7 min), is
//! not a candidate, and is never kept. `Results::control` records its config and `Quality`.
//!
//! A KLD run *fails* when llama-perplexity exits non-zero, times out, or prints no parsable
//! `Mean KLD` / `Same top p` (`bench::kld_outcome`). What a failure means, by step:
//! - baseline logits (`kld-0`): the sweep exits `FAILED` naming `<out>/kld-0.log`; no candidate is
//!   benched (none could be judged, and each costs ~7 min).
//! - control (`kld-control`): the sweep exits `FAILED` naming `<out>/kld-control.log`, same reason;
//!   so `Results::control` always carries a reading.
//! - candidate `i` (`kld-<i>`): that candidate is `Crashed` with `Point::failure` =
//!   `kld: <exit N | timed out | no Mean KLD / Same top p in output>: <last non-empty log line>`.
//!   A measurement failure is never reported as a quality verdict.
//!
//! Every KLD run's full output is kept, written the moment the run returns (so the log exists even
//! when the sweep then exits `FAILED`): `<out>/kld-0.log` (baseline logits), `<out>/kld-control.log`,
//! `<out>/kld-<i>.log` for plan point `i` (= `points[i-1]`; absent iff that candidate's bench
//! failed, since no KLD was attempted). `<out>` is created before the first remote process starts.
//! The casebook draft states the control's config and KLD and the `-c <kld_ctx> --chunks <n>` used.
//!
//! # Keep rule (`decide`), in order
//! 1. `Crashed`: `failure` set, any depth in `workload.depths` with fewer than `repeats` samples, or
//!    no KLD reading (`kld` null without a `failure`: a measurement that did not happen is not a
//!    quality result). Detail names the failure, or says `no KLD reading`.
//! 2. `IdenticalConfig`: candidate config equals the baseline's.
//! 3. `TooFewSamples`: any series of `Workload::series` has fewer than `thresholds.min_samples`
//!    samples on either side.
//! 4. `QualityGate`, relative to the control's `Quality`. First `control.mean_kld >
//!    control_kld_cap` rejects with detail `control KLD <c:.4> > cap <cap:.4>` (a broken instrument
//!    cannot wave anything through). Then `mean_kld > control.mean_kld + kld_margin` rejects with
//!    detail `KLD <q:.4> > control <c:.4> + <margin:.4>`, and `same_top_pct < control.same_top_pct -
//!    same_top_margin_pct` with `same-top <q:.1>% < control <c:.1>% - <margin:.1>%`. Exactly at a
//!    bound passes.
//! 5. `GuardRegression`: any series of `Workload::guarded` (every series but the objective) has
//!    `delta_pct = (mean_c - mean_b) / mean_b * 100 < -guard_pct`. Exactly `-guard_pct` passes.
//!    `detail` lists every violator in `Series` order as `<series> <delta:+.1>% < -<guard:.1>%`,
//!    joined by `; ` (e.g. `tg@0 -10.3% < -3.0%; tg@32768 -7.0% < -3.0%`).
//! 6. `NoiseGate`, on the objective series only: Welch lower bound
//!    `delta - noise_k * sqrt(s_b^2/n_b + s_c^2/n_c) <= 0`, where `delta = mean_c - mean_b` in
//!    tok/s and `s` is the sample stdev (n-1).
//! 7. `Cleared`: keep.
//!
//! Quality first, then the guard, then noise: a candidate is never kept for a gain it bought with
//! a regression elsewhere, and a tradeoff the operator does want is stated as a larger `guard_pct`.

use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

mod fixture;
mod workload;
pub use fixture::{DecideFixture, DecideOutput, Expectation, FixtureBaseline, FixtureCandidate};
pub use workload::{Depth, Metric, Series, SeriesDelta, SeriesStats, Workload};

/// 3: `Results` and `DecideFixture` carry the control's quality, `Thresholds` are margins over it,
/// `Workload` has `kld_ctx` (v1.1). 2: per-series stats and `workload` (v1).
pub const SCHEMA_VERSION: u32 = 3;
/// Headroom that must remain after the model is resident (strix-casebook convention: 20 GiB).
pub const DEFAULT_RESERVE_GIB: f64 = 20.0;
/// Vocabulary size assumed for the logits-file estimate when the GGUF header has no
/// `tokenizer.ggml.tokens`: 262144 (Gemma-class), the largest in common use, so the estimate errs high.
pub const FALLBACK_N_VOCAB: u64 = 262_144;

/// Process exit codes. The harness tells a refusal from a crash by these.
pub mod exit {
    pub const OK: i32 = 0;
    /// Ran, but failed (bench crashed, ssh died, bad spec).
    pub const FAILED: i32 = 1;
    /// clap usage error.
    pub const USAGE: i32 = 2;
    /// Preflight refused; nothing was started on the host.
    pub const REFUSED: i32 = 3;
    /// Subcommand exists but has no implementation yet.
    pub const UNIMPLEMENTED: i32 = 70;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Host {
    Strix,
    Spark,
    Spark2,
}

impl Host {
    /// GB10 unified memory: a model loaded beside a resident one OOM-zombies the box.
    pub fn is_spark(self) -> bool {
        match self {
            Host::Strix => false,
            Host::Spark | Host::Spark2 => true,
        }
    }
}

impl fmt::Display for Host {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Host::Strix => "strix",
            Host::Spark => "spark",
            Host::Spark2 => "spark2",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// `~/src/llama.cpp/build-vulkan/bin`
    Vulkan,
    /// `~/src/llama.cpp/build-hip/bin`
    Hip,
}

/// llama.cpp `-ctk` / `-ctv` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    F16,
    Q8_0,
    Q4_0,
}

/// One llama.cpp configuration. Field order is the sort order of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub backend: Backend,
    /// `-fa`
    pub fa: bool,
    /// `-ub`
    pub ub: u32,
    /// `-b`
    pub b: u32,
    pub ctk: KvType,
    pub ctv: KvType,
    /// `-c`
    pub ctx: u32,
    /// `-np`
    pub np: u32,
    /// `ROCBLAS_USE_HIPBLASLT=1`; meaningless (normalized to false) on Vulkan.
    pub hipblaslt: bool,
}

/// Tokens per second.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TokS(pub f64);

/// Gibibytes (2^30 bytes).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Gib(pub f64);

/// Mean KL divergence of a candidate's token distributions from the baseline's.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Kld(pub f64);

/// A percentage, 0..=100 scale.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Pct(pub f64);

/// One repeat of llama-bench at one depth: pp512 and tg128 throughput.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    /// v0 files carry no depth; they were measured at 0.
    #[serde(default)]
    pub depth: Depth,
    pub pp_tok_s: TokS,
    pub tg_tok_s: TokS,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stats {
    pub mean: TokS,
    /// Sample standard deviation (n-1).
    pub stdev: TokS,
    pub n: u32,
}

/// llama-perplexity `--kl-divergence` against the baseline's saved logits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quality {
    pub mean_kld: Kld,
    pub same_top_pct: Pct,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Keep,
    Reject,
    /// The anchor itself; never kept or rejected.
    Baseline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonKind {
    Baseline,
    Crashed,
    IdenticalConfig,
    TooFewSamples,
    QualityGate,
    /// A guarded (non-objective) series regressed more than `Workload::guard_pct`.
    GuardRegression,
    NoiseGate,
    Cleared,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reason {
    pub kind: ReasonKind,
    /// Numbers behind the verdict, for a human (e.g. "KLD 0.070 > 0.050").
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Decision {
    pub verdict: Verdict,
    pub reason: Reason,
}

/// Keep-rule thresholds. Defaults are the fleet values; `decide` fixtures carry their own copy
/// so the file documents what it tests.
///
/// - `kld_margin` 0.02 and `same_top_margin_pct` 3.0, over the reorder-only control (v1.1; the v1
///   absolute `kld_max` 0.05 / `same_top_min_pct` 85 rejected a bare `-ub` change at 32k). Measured
///   at 32k (strix-casebook 2026-10-21-tuner-v1-depth-sweep §3): control 0.079 / 82.4%, f16 K +
///   q8_0 V +0.007 / -1.0 pts, q8_0/q8_0 +0.017 / -2.5; at 512 (§2): control 0.032 / 87.7%,
///   q8_0/q8_0 +0.015 / -2.6, ub 2048 q8_0/q8_0 +0.019 / -2.9. A task-level check (§4) found q8_0 V
///   equal to f16 (47/48 both arms), so all of those must pass; the W4A4 regression (0.11 / 74%
///   against a ~0.035 / 87% control: +0.07 / -13 pts) must fail. 0.02 and 3.0 sit just above the
///   worst measured honest case (+0.019 / -2.9).
/// - `control_kld_cap` 0.2: the largest honest control drift measured is 0.079 (32k); 0.2 is 2.5x
///   that. Above it the instrument is broken (wrong logits file, wrong dataset, a crashed kernel
///   path), not the candidates, so nothing can be judged. It is not a quality threshold: W4A4's 0.11
///   is under it on purpose.
/// - `noise_k` 2.0 on the Welch standard error: one-sided ~95% for large n; with `min_samples` 3
///   the true t critical is ~2.1-2.9, so this is slightly permissive and cheap on bench time.
/// - `min_samples` 3: a stdev needs n >= 3 to mean anything.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Thresholds {
    pub kld_margin: Kld,
    pub same_top_margin_pct: Pct,
    pub control_kld_cap: Kld,
    pub noise_k: f64,
    pub min_samples: u32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            kld_margin: Kld(0.02),
            same_top_margin_pct: Pct(3.0),
            control_kld_cap: Kld(0.2),
            noise_k: 2.0,
            min_samples: 3,
        }
    }
}

/// The reorder-only control of a sweep (module docs, "Quality"): the baseline with only `ub`
/// changed, and its KLD against the baseline logits. Measured, never benched, never kept.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Control {
    pub config: Config,
    pub kld: Quality,
}

/// One measured configuration in `results.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Point {
    pub config: Config,
    /// Repeats requested per depth; fewer samples than this at any depth means a crash.
    pub repeats: u32,
    /// One per (repeat, depth), depth-major; see the module docs.
    pub samples: Vec<Sample>,
    /// Set when a bench repeat crashed or timed out (exit status / last stderr line), or when the
    /// KLD run failed (`kld: ...`, module docs "Quality").
    pub failure: Option<String>,
    /// One entry per series that has samples, in `Series` order; empty iff `samples` is.
    pub series: Vec<SeriesStats>,
    /// `None` for the baseline and for crashed points (bench or KLD); never `None` on a measured
    /// candidate without a `failure`.
    pub kld: Option<Quality>,
    /// tg@0 mean as % of `Results::roofline.expected_tg_tok_s`; `None` when tg@0 was not measured.
    pub roofline_pct: Option<Pct>,
    pub decision: Decision,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub path: PathBuf,
    pub bytes: u64,
}

/// What every casebook number must state (strix-casebook CLAUDE.md).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stack {
    /// `git describe --tags` + short hash, e.g. "v0.6.0 d81235049384".
    pub llama_cpp: String,
    pub kernel: String,
    /// `/sys/firmware/acpi/platform_profile`.
    pub platform_profile: String,
    /// Mesa (Vulkan) or ROCm (HIP) version string of the baseline's backend.
    pub driver: String,
}

/// Roofline inputs via the `roofline` crate (`roofline::roofline_tok_s(hw, 1, active_gb)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Roofline {
    /// Id in roofline-labs/benchmarks/data/hardware.toml.
    pub hardware_id: String,
    pub decode_bw_gbs: f64,
    pub active_gb_per_token: f64,
    pub expected_tg_tok_s: TokS,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Results {
    pub schema_version: u32,
    pub host: Host,
    pub model: ModelRef,
    pub stack: Stack,
    pub roofline: Roofline,
    /// RFC 3339 UTC.
    pub started_utc: String,
    pub baseline: Point,
    /// The reorder-only control every candidate's quality is judged against.
    pub control: Control,
    /// Candidates, in plan order.
    pub points: Vec<Point>,
}

/// The recipe store entry: `<recipes>/<host>/<model stem>.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Recipe {
    pub schema_version: u32,
    pub host: Host,
    /// GGUF file name (not path), e.g. "gpt-oss-120b-MXFP4.gguf".
    pub model: String,
    /// The kept config with the highest objective mean, or the baseline when nothing was kept.
    pub winner: Config,
    /// What the winner won at (`Workload::objective` of the sweep).
    pub objective: Series,
    /// The winner's measured series (`Point::series`).
    pub series: Vec<SeriesStats>,
    pub recorded_utc: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Baseline,
    WarmStart,
    Grid,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanPoint {
    pub role: Role,
    pub config: Config,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WarmStart {
    pub recipe: PathBuf,
    pub config: Config,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema_version: u32,
    pub host: Host,
    pub model: PathBuf,
    pub repeats: u32,
    /// The spec's workload, defaults filled in.
    pub workload: Workload,
    pub warm_start: Option<WarmStart>,
    /// `points[0]` is the baseline; see the module docs for the rest of the order.
    pub points: Vec<PlanPoint>,
}

/// Axes of the sweep; each listed axis is a list of values to try.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grid {
    pub backend: Option<Vec<Backend>>,
    pub fa: Option<Vec<bool>>,
    pub ub: Option<Vec<u32>>,
    pub b: Option<Vec<u32>>,
    pub ctk: Option<Vec<KvType>>,
    pub ctv: Option<Vec<KvType>>,
    pub ctx: Option<Vec<u32>>,
    pub np: Option<Vec<u32>>,
    pub hipblaslt: Option<Vec<bool>>,
}

/// `sweep.toml`. `plan`/`sweep` reject (exit `FAILED`, naming the field) a spec whose
/// `workload.objective.depth` is not in `workload.depths`, whose `depths` is empty or repeats a
/// value, whose `guard_pct` is negative, or whose `baseline` has no reorder-only control
/// (`bench::control_config` is `None`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spec {
    pub schema_version: u32,
    pub repeats: u32,
    #[serde(default)]
    pub workload: Workload,
    pub baseline: Config,
    #[serde(default)]
    pub grid: Grid,
}

