# roofline-benchmarks

Measured LLM decode speed — model × hardware × engine × quant → tok/s — with each number scored
against the hardware's **memory-bandwidth roofline**, so you can tell "this model is slow" from
"this setup is broken".

- [`TABLE.md`](TABLE.md) — the rendered table (generated).
- [`SCHEMA.md`](SCHEMA.md) — field reference and how to submit a row.
- [`MEASURE-LATER.md`](MEASURE-LATER.md) — known gaps.
- `data/` — the source of truth: `hardware.toml`, `models.toml`, `runs/*.toml`.

```
cargo run -- validate            # schema, foreign keys, sanity rules
cargo run -- render > TABLE.md   # regenerate the table
cargo test
```

Rows currently come from one lab's DGX Spark (GB10) notebook, May–September 2026. Numbers are
from serving benchmarks under stated configs, not vendor claims; anything third-party or
unreproduced is excluded. License: CC-BY-4.0 for data, MIT for code.
