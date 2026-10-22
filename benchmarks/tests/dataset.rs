use roofline_benchmarks::{Dataset, roofline_ratio};
use std::path::Path;

fn shipped() -> Dataset {
    let ds = Dataset::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("data").as_path()).unwrap();
    assert!(ds.validate().is_empty(), "{:#?}", ds.validate());
    ds
}

#[test]
fn anchor_row_is_exactly_100_percent() {
    let ds = shipped();
    let run = ds.runs.iter().find(|r| r.id == "2026-08-19-ornith-q8-llamacpp-c1").unwrap();
    let pct = roofline_ratio(ds.hardware("gb10").unwrap(), run).unwrap() * 100.0;
    assert!((pct - 100.0).abs() < 0.5, "anchor drifted: {pct}");
}

#[test]
fn two_node_cell_scales_the_divisor() {
    let ds = shipped();
    let run = ds.runs.iter().find(|r| r.id == "2026-09-07-minimax-m27-awq-vllm-tp2-c1").unwrap();
    let ratio = roofline_ratio(ds.hardware("gb10").unwrap(), run).unwrap();
    // 41.9 / (2 x 187 / 5.92) = 0.663
    assert!((ratio - 0.663).abs() < 0.005, "{ratio}");
}

#[test]
fn rows_without_active_bytes_have_no_ratio() {
    let ds = shipped();
    let run = ds.runs.iter().find(|r| r.id == "2026-09-06-mistral-small-4-nvfp4-vllm-c1").unwrap();
    assert!(roofline_ratio(ds.hardware("gb10").unwrap(), run).is_none());
}

#[test]
fn validation_catches_broken_rows() {
    let dir = std::env::temp_dir().join(format!("rb-test-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("data/hardware.toml"),
        dir.join("hardware.toml"),
    )
    .unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("data/models.toml"),
        dir.join("models.toml"),
    )
    .unwrap();
    std::fs::write(
        dir.join("runs/bad.toml"),
        r#"
[[run]]
id = "bad"
date = "2026-9-1"
model = "no-such-model"
checkpoint = "x"
hardware = "gb10"
nodes = 2
engine = "vllm"
engine_version = "1"
quant = "Q8_0"
active_gb_per_token = 3.0
speculation = "none"
concurrency = 1
decode_tok_s = -1.0
decode_per_stream_tok_s = 5.0
prefill_tok_s = 100.0
source = "lab:test"

[[run]]
id = "bad"
date = "2026-09-01"
model = "gpt-oss-120b"
checkpoint = "x"
hardware = "gb10"
engine = "vllm"
engine_version = "1"
quant = "Q8_0"
speculation = "none"
concurrency = 1
decode_tok_s = 1.0
source = "lab:test"
"#,
    )
    .unwrap();
    let ds = Dataset::load(&dir).unwrap();
    let errs = ds.validate();
    std::fs::remove_dir_all(&dir).unwrap();
    let joined = errs.join("\n");
    for needle in [
        "duplicate run id",
        "unknown model id",
        "not YYYY-MM-DD",
        "requires parallelism",
        "decode_tok_s must be > 0",
        "requires active_gb_method",
        "only for concurrency > 1",
        "go together",
    ] {
        assert!(joined.contains(needle), "missing {needle:?} in:\n{joined}");
    }
}

#[test]
fn unknown_field_is_rejected() {
    let dir = std::env::temp_dir().join(format!("rb-test-uf-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("runs")).unwrap();
    std::fs::write(dir.join("hardware.toml"), "hardware = []\n").unwrap();
    std::fs::write(dir.join("models.toml"), "model = []\n").unwrap();
    std::fs::write(dir.join("runs/a.toml"), "[[run]]\nid=\"a\"\ntokens_per_sec = 1\n").unwrap();
    let err = Dataset::load(&dir).unwrap_err();
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(err.contains("unknown field") || err.contains("missing field"), "{err}");
}

#[test]
fn html_page_has_one_row_per_run_and_the_anchor_at_100() {
    let ds = shipped();
    let page = roofline_benchmarks::html::page(&ds);
    assert_eq!(page.matches("<tr class=\"run\"").count(), ds.runs.len());
    assert!(page.contains("<b>100%</b><span>at ceiling</span>"), "anchor bar missing");
    assert!(!page.contains("<script src"), "page must be self-contained");
}
