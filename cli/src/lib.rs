//! Bytes read per decoded token, derived from a model's architecture and a quant's bytes/param.
//! The formula itself (`nodes x decode_bw / active GB) lives in roofline-benchmarks.

use roofline_benchmarks::{Hardware, Shape};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Quant {
    pub id: String,
    /// Attention, shared experts, embeddings, output head.
    pub bytes_per_param: f64,
    /// Routed experts.
    pub expert_bytes_per_param: f64,
    pub notes: Option<String>,
}

#[derive(Deserialize)]
pub struct QuantFile {
    pub quant: Vec<Quant>,
}

/// Router weights stay F32 in GGUF conversions, whatever the quant.
const ROUTER_BYTES: f64 = 4.0;

/// Parameter counts split by how they are read at decode time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    /// Attention + shared experts: read every token, stored at the dense rate.
    pub dense: u64,
    /// Expert router: read every token, F32.
    pub router: u64,
    /// Input embedding table: one row per token, so never in the active count.
    pub embedding: u64,
    /// Output head: read every token unless tied to the embedding (then still read, but not
    /// stored twice).
    pub head: u64,
    pub experts_total: u64,
    pub experts_active: u64,
}

impl Params {
    pub fn total(&self) -> u64 {
        let emb = self.embedding;
        self.dense + self.router + emb + self.head + self.experts_total
    }

    pub fn active(&self) -> u64 {
        self.dense + self.router + self.head + self.experts_active
    }
}

pub fn count(s: &Shape) -> Params {
    let h = s.hidden;
    let q_width = s.n_heads * s.head_dim * if s.attn_output_gate { 2 } else { 1 };
    let full = h * q_width + 2 * h * s.n_kv_heads * s.head_dim + s.n_heads * s.head_dim * h;
    let linear_layers = s.layers - s.full_attention_layers;
    let linear = s.linear_attention.map_or(0, |la| {
        let k = la.key_heads * la.key_head_dim;
        let v = la.value_heads * la.value_head_dim;
        // in_proj (q, k, v, z), in_proj (b, a), conv1d, out_proj
        h * (2 * k + 2 * v) + h * 2 * la.value_heads + (2 * k + v) * 4 + v * h
    });
    let mut dense = s.full_attention_layers * full + linear_layers * linear;
    let (router, experts_total, experts_active) = match s.moe {
        Some(m) => {
            let expert = 3 * h * m.intermediate;
            dense += s.layers * (3 * h * m.shared_intermediate + h);
            (s.layers * m.n_experts * h, s.layers * m.n_experts * expert, s.layers * m.top_k * expert)
        }
        None => {
            dense += s.layers * 3 * h * s.mlp_intermediate;
            (0, 0, 0)
        }
    };
    let head = s.vocab * h;
    Params {
        dense,
        router,
        embedding: if s.tied_embeddings { 0 } else { head },
        head,
        experts_total,
        experts_active,
    }
}

/// Bytes read per decoded token, in decimal GB.
pub fn active_gb(p: &Params, q: &Quant) -> f64 {
    let dense = (p.dense + p.head) as f64 * q.bytes_per_param;
    let router = p.router as f64 * ROUTER_BYTES;
    let experts = p.experts_active as f64 * q.expert_bytes_per_param;
    (dense + router + experts) / 1e9
}

/// Resident weight bytes, in decimal GB.
pub fn weights_gb(p: &Params, q: &Quant) -> f64 {
    let dense = (p.dense + p.head + p.embedding) as f64 * q.bytes_per_param;
    let router = p.router as f64 * ROUTER_BYTES;
    let experts = p.experts_total as f64 * q.expert_bytes_per_param;
    (dense + router + experts) / 1e9
}

/// KV cache for `ctx` tokens at `kv_bytes` per element: only softmax-attention layers grow with
/// context; linear-attention state is fixed-size and ignored here.
pub fn kv_gb(s: &Shape, ctx: u64, kv_bytes: f64) -> f64 {
    (2 * s.full_attention_layers * s.n_kv_heads * s.head_dim * ctx) as f64 * kv_bytes / 1e9
}

pub fn roofline_tok_s(hw: &Hardware, nodes: u32, active_gb: f64) -> f64 {
    f64::from(nodes) * hw.decode_bw_gbs / active_gb
}

pub fn memory_gb(hw: &Hardware, nodes: u32) -> f64 {
    f64::from(nodes) * hw.memory_gib * 1.073_741_824
}
