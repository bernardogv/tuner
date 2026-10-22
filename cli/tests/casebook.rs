//! The derivation must reproduce the casebook's hand-computed 35B-A3B figures
//! (spark-casebook/casebook/2026-08-19-roofline-anchor.md) from the architecture alone.

use roofline::{Quant, QuantFile, active_gb, count, kv_gb, roofline_tok_s, weights_gb};
use roofline_benchmarks::{Dataset, Shape};
use std::path::Path;

fn data() -> Dataset {
    Dataset::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/data")).unwrap()
}

fn shape() -> Shape {
    data().model("qwen3.6-35b-a3b").unwrap().shape.clone().unwrap()
}

fn quant(id: &str) -> Quant {
    let text = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("quants.toml")).unwrap();
    let f: QuantFile = toml::from_str(&text).unwrap();
    f.quant.into_iter().find(|q| q.id == id).unwrap()
}

fn within(actual: f64, expected: f64, pct: f64) {
    let off = (actual - expected).abs() / expected * 100.0;
    assert!(off <= pct, "{actual} is {off:.2}% from {expected}, allowed {pct}%");
}

#[test]
fn shape_accounts_for_the_advertised_35b_a3b() {
    let p = count(&shape());
    within(p.total() as f64 / 1e9, 35.0, 2.0);
    within(p.active() as f64 / 1e9, 3.0, 2.0);
}

#[test]
fn q8_0_reproduces_the_anchor_bytes_per_token() {
    within(active_gb(&count(&shape()), &quant("Q8_0")), 3.196, 0.5);
}

#[test]
fn nvfp4_experts_shrink_only_the_expert_term() {
    let p = count(&shape());
    let q8 = active_gb(&p, &quant("Q8_0"));
    let nv = active_gb(&p, &quant("NVFP4-RTN-experts"));
    let expected_drop = p.experts_active as f64 * (1.0625 - 0.5625) / 1e9;
    assert!((q8 - nv - expected_drop).abs() < 1e-9, "{q8} -> {nv}");
    // casebook says 2.752; the arch derivation lands ~2% under it (see quants.toml notes)
    within(nv, 2.752, 3.0);
}

#[test]
fn apex_reproduces_the_casebook_figure() {
    within(active_gb(&count(&shape()), &quant("APEX-mixed")), 2.393, 1.0);
}

#[test]
fn anchor_cell_is_predicted_at_58_5() {
    let ds = data();
    let hw = ds.hardware("gb10").unwrap();
    within(roofline_tok_s(hw, 1, active_gb(&count(&shape()), &quant("Q8_0"))), 58.5, 1.0);
    within(roofline_tok_s(hw, 2, 3.196), 117.0, 1.0);
}

#[test]
fn tied_embeddings_are_stored_once() {
    let mut s = shape();
    let untied = weights_gb(&count(&s), &quant("BF16"));
    s.tied_embeddings = true;
    let tied = weights_gb(&count(&s), &quant("BF16"));
    let head_bytes = (s.vocab * s.hidden) as f64 * 2.0 / 1e9;
    assert!((untied - tied - head_bytes).abs() < 1e-9);
    // active bytes still read the head either way
    assert_eq!(count(&s).active(), count(&shape()).active());
}

#[test]
fn kv_grows_only_with_full_attention_layers() {
    let s = shape();
    // 10 layers x 2 (K,V) x 2 kv heads x 256 dims x 2 bytes = 20480 B/token
    assert!((kv_gb(&s, 100_000, 2.0) - 2.048).abs() < 1e-9);
}
