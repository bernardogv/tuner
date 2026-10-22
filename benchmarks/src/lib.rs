//! Schema, loading and validation for the benchmark dataset, plus the roofline arithmetic.
//! Field semantics are documented for contributors in SCHEMA.md; keep the two in sync.

pub mod html;
pub mod render;

use serde::Deserialize;
use std::fmt;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Hardware {
    pub id: String,
    pub name: String,
    pub memory_gib: f64,
    pub memory_kind: Option<String>,
    pub spec_bw_gbs: f64,
    pub stream_copy_gbs: f64,
    /// Achieved bandwidth of real single-stream decode, GB/s. The roofline divisor.
    pub decode_bw_gbs: f64,
    pub bf16_tflops: Option<f64>,
    pub notes: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub arch: String,
    pub params_total_b: f64,
    pub params_active_b: Option<f64>,
    pub notes: Option<String>,
    /// Architecture dimensions; lets the calculator derive bytes/token instead of trusting
    /// `params_active_b`. Optional — rows validate without it.
    pub shape: Option<Shape>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shape {
    pub hidden: u64,
    pub layers: u64,
    pub vocab: u64,
    #[serde(default)]
    pub tied_embeddings: bool,
    /// Softmax-attention layers; the remaining `layers` are linear attention.
    pub full_attention_layers: u64,
    pub n_heads: u64,
    pub n_kv_heads: u64,
    pub head_dim: u64,
    /// Qwen3-Next style gated attention: q_proj is twice as wide.
    #[serde(default)]
    pub attn_output_gate: bool,
    pub linear_attention: Option<LinearAttention>,
    /// Absent means a dense MLP of `mlp_intermediate` on every layer.
    pub moe: Option<Moe>,
    #[serde(default)]
    pub mlp_intermediate: u64,
}

/// Gated DeltaNet dims (Qwen3-Next / Qwen3.5 hybrids).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinearAttention {
    pub key_heads: u64,
    pub key_head_dim: u64,
    pub value_heads: u64,
    pub value_head_dim: u64,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Moe {
    pub n_experts: u64,
    pub top_k: u64,
    pub intermediate: u64,
    #[serde(default)]
    pub shared_intermediate: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Speculation {
    None,
    DraftMtp,
    Nextn,
    Mtp,
    Dflash,
    Unknown,
}

impl fmt::Display for Speculation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Speculation::None => "off",
            Speculation::DraftMtp => "draft-mtp",
            Speculation::Nextn => "nextn",
            Speculation::Mtp => "mtp",
            Speculation::Dflash => "dflash",
            Speculation::Unknown => "?",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActiveGbMethod {
    Arch,
    Fraction,
    Assumed,
}

fn one() -> u32 {
    1
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Run {
    pub id: String,
    pub date: String,
    pub model: String,
    pub checkpoint: String,
    pub hardware: String,
    #[serde(default = "one")]
    pub nodes: u32,
    pub parallelism: Option<String>,
    pub engine: String,
    pub engine_version: String,
    pub quant: String,
    pub checkpoint_gb: Option<f64>,
    pub active_gb_per_token: Option<f64>,
    pub active_gb_method: Option<ActiveGbMethod>,
    pub speculation: Speculation,
    pub concurrency: u32,
    /// Per-stream at concurrency 1, aggregate above.
    pub decode_tok_s: f64,
    pub decode_per_stream_tok_s: Option<f64>,
    pub context_tokens: Option<u64>,
    pub prefill_tok_s: Option<f64>,
    pub prefill_prompt_tokens: Option<u64>,
    pub config: Option<String>,
    pub source: String,
    pub notes: Option<String>,
}

#[derive(Debug, Default)]
pub struct Dataset {
    pub hardware: Vec<Hardware>,
    pub models: Vec<Model>,
    pub runs: Vec<Run>,
}

#[derive(Deserialize)]
struct HardwareFile {
    hardware: Vec<Hardware>,
}
#[derive(Deserialize)]
struct ModelFile {
    model: Vec<Model>,
}
#[derive(Deserialize)]
struct RunFile {
    run: Vec<Run>,
}

fn read_toml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

impl Dataset {
    /// Load `hardware.toml`, `models.toml` and every `runs/*.toml` under `dir`.
    pub fn load(dir: &Path) -> Result<Dataset, String> {
        let hw: HardwareFile = read_toml(&dir.join("hardware.toml"))?;
        let models: ModelFile = read_toml(&dir.join("models.toml"))?;
        let runs_dir = dir.join("runs");
        let mut paths: Vec<_> = fs::read_dir(&runs_dir)
            .map_err(|e| format!("{}: {e}", runs_dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        paths.sort();
        let mut runs = Vec::new();
        for p in paths {
            let f: RunFile = read_toml(&p)?;
            runs.extend(f.run);
        }
        Ok(Dataset { hardware: hw.hardware, models: models.model, runs })
    }

    pub fn hardware(&self, id: &str) -> Option<&Hardware> {
        self.hardware.iter().find(|h| h.id == id)
    }

    pub fn model(&self, id: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Every schema rule a contributor can break. Returns all problems, not just the first.
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        check_unique("hardware", self.hardware.iter().map(|h| h.id.as_str()), &mut errs);
        check_unique("model", self.models.iter().map(|m| m.id.as_str()), &mut errs);
        check_unique("run", self.runs.iter().map(|r| r.id.as_str()), &mut errs);
        for h in &self.hardware {
            for (name, v) in [
                ("memory_gib", h.memory_gib),
                ("spec_bw_gbs", h.spec_bw_gbs),
                ("stream_copy_gbs", h.stream_copy_gbs),
                ("decode_bw_gbs", h.decode_bw_gbs),
            ] {
                if v <= 0.0 {
                    errs.push(format!("hardware {}: {name} must be > 0", h.id));
                }
            }
            if h.decode_bw_gbs > h.stream_copy_gbs || h.stream_copy_gbs > h.spec_bw_gbs {
                errs.push(format!("hardware {}: expect decode_bw <= stream_copy <= spec_bw", h.id));
            }
        }
        for m in &self.models {
            if let Some(s) = &m.shape {
                if s.full_attention_layers > s.layers {
                    errs.push(format!("model {}: full_attention_layers > layers", m.id));
                }
                if s.full_attention_layers < s.layers && s.linear_attention.is_none() {
                    errs.push(format!("model {}: linear layers need [linear_attention]", m.id));
                }
                match s.moe {
                    Some(moe) if moe.top_k == 0 || moe.top_k > moe.n_experts => {
                        errs.push(format!("model {}: expect 1 <= top_k <= n_experts", m.id));
                    }
                    None if s.mlp_intermediate == 0 => {
                        errs.push(format!("model {}: need moe or mlp_intermediate", m.id));
                    }
                    _ => {}
                }
            }
        }
        for r in &self.runs {
            let e = |msg: String| format!("run {}: {msg}", r.id);
            if self.hardware(&r.hardware).is_none() {
                errs.push(e(format!("unknown hardware id {:?}", r.hardware)));
            }
            if self.model(&r.model).is_none() {
                errs.push(e(format!("unknown model id {:?}", r.model)));
            }
            if !is_iso_date(&r.date) {
                errs.push(e(format!("date {:?} is not YYYY-MM-DD", r.date)));
            }
            if r.nodes == 0 || r.concurrency == 0 {
                errs.push(e("nodes and concurrency must be >= 1".into()));
            }
            if r.nodes > 1 && r.parallelism.is_none() {
                errs.push(e("nodes > 1 requires parallelism".into()));
            }
            if r.decode_tok_s <= 0.0 {
                errs.push(e("decode_tok_s must be > 0".into()));
            }
            match (r.active_gb_per_token, r.active_gb_method) {
                (Some(g), Some(_)) if g <= 0.0 => errs.push(e("active_gb_per_token must be > 0".into())),
                (Some(_), None) => errs.push(e("active_gb_per_token requires active_gb_method".into())),
                (None, Some(_)) => errs.push(e("active_gb_method without active_gb_per_token".into())),
                _ => {}
            }
            if r.concurrency == 1 && r.decode_per_stream_tok_s.is_some() {
                errs.push(e("decode_per_stream_tok_s is only for concurrency > 1".into()));
            }
            if let Some(ps) = r.decode_per_stream_tok_s
                && ps > r.decode_tok_s
            {
                errs.push(e("per-stream tok/s exceeds aggregate".into()));
            }
            if r.prefill_tok_s.is_some() != r.prefill_prompt_tokens.is_some() {
                errs.push(e("prefill_tok_s and prefill_prompt_tokens go together".into()));
            }
            if r.source.is_empty() {
                errs.push(e("source is required".into()));
            }
        }
        errs
    }
}

fn check_unique<'a>(kind: &str, ids: impl Iterator<Item = &'a str>, errs: &mut Vec<String>) {
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        if !seen.insert(id) {
            errs.push(format!("duplicate {kind} id {id:?}"));
        }
    }
}

fn is_iso_date(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 10
        && b.iter().enumerate().all(|(i, c)| match i {
            4 | 7 => *c == b'-',
            _ => c.is_ascii_digit(),
        })
}

/// Expected single-stream decode with no speculation: bytes/s the hardware actually delivers,
/// divided by bytes read per token. Multi-node cells get `nodes x` the bandwidth.
pub fn roofline_tok_s(hw: &Hardware, run: &Run) -> Option<f64> {
    run.active_gb_per_token
        .map(|gb| f64::from(run.nodes) * hw.decode_bw_gbs / gb)
}

/// measured / roofline. Above 1.0 means speculation, batching, or a better-than-anchor kernel path.
pub fn roofline_ratio(hw: &Hardware, run: &Run) -> Option<f64> {
    roofline_tok_s(hw, run).map(|r| run.decode_tok_s / r)
}
