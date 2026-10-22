# Schema and submission format

Three TOML files, one Rust binary that validates and renders them. Contributors add a file under
`data/runs/`; `cargo run -- validate` must pass before a row is merged.

## The one formula

```
roofline tok/s  =  nodes × hardware.decode_bw_gbs ÷ run.active_gb_per_token
% roofline      =  run.decode_tok_s ÷ roofline tok/s
```

`decode_bw_gbs` is **achieved** bandwidth, not the spec sheet: the bytes/s a real single-stream
decode with no speculation actually pulls on that hardware (measured tok/s × active GB/token on a
clean cell). On GB10 that is 187 GB/s = 76% of STREAM copy = 68% of the spec sheet. The table's
100% therefore means "as good as the anchor cell", and it is possible to beat it: batched vLLM and
SGLang decode land at ~110% (≈85% of STREAM), speculation at 160–185%.

`active_gb_per_token` is the bytes read per decoded token — for a MoE, dense layers + output head
+ (experts × top-k ÷ n-experts), at the quant's bytes/param. Compute it from the architecture when
you can (`active_gb_method = "arch"`); otherwise `checkpoint_gb × active_params ÷ total_params`
(`"fraction"`), which over-counts slightly because attention is usually kept at higher precision.

## `data/hardware.toml` — `[[hardware]]`

| field | type | required | meaning |
|---|---|---|---|
| `id` | string | yes | short slug, referenced by runs (`gb10`) |
| `name` | string | yes | |
| `memory_gib` | float | yes | total memory available to the model |
| `memory_kind` | string | no | unified/discrete, swap state |
| `spec_bw_gbs` | float | yes | vendor spec-sheet bandwidth |
| `stream_copy_gbs` | float | yes | measured STREAM/BabelStream copy bandwidth |
| `decode_bw_gbs` | float | yes | achieved single-stream decode bandwidth — the roofline divisor. Must be ≤ `stream_copy_gbs` ≤ `spec_bw_gbs` |
| `bf16_tflops` | float | no | for prefill context only |
| `notes` | string | no | how `decode_bw_gbs` was established |

## `data/models.toml` — `[[model]]`

One entry per *architecture*; checkpoints and quants live on the run.

| field | type | required |
|---|---|---|
| `id` | string | yes |
| `name` | string | yes |
| `arch` | string | yes — layers, experts, top-k, attention type, MTP head |
| `params_total_b` | float | yes |
| `params_active_b` | float | no — leave out if you cannot derive it; the run's `%` becomes `—` |
| `notes` | string | no |
| `shape` | table | no — architecture dims for the calculator (`cli/`), see below |

### `[model.shape]`

Lets `cli/` derive bytes/token from the architecture instead of trusting `params_active_b`.
Unknown fields are rejected.

| field | type | required | meaning |
|---|---|---|---|
| `hidden`, `layers`, `vocab` | int | yes | |
| `tied_embeddings` | bool | no (false) | output head shares the embedding table |
| `full_attention_layers` | int | yes | softmax-attention layers; the rest are linear attention |
| `n_heads`, `n_kv_heads`, `head_dim` | int | yes | |
| `attn_output_gate` | bool | no (false) | Qwen3-Next gated attention (q_proj twice as wide) |
| `linear_attention` | inline table | if `full_attention_layers < layers` | `{ key_heads, key_head_dim, value_heads, value_head_dim }` (Gated DeltaNet) |
| `moe` | inline table | one of `moe` / `mlp_intermediate` | `{ n_experts, top_k, intermediate, shared_intermediate }` |
| `mlp_intermediate` | int | | dense MLP width when there is no `moe` |

## `data/runs/<contributor>.toml` — `[[run]]`

| field | type | required | meaning |
|---|---|---|---|
| `id` | string | yes | unique; convention `YYYY-MM-DD-<model>-<quant>-<engine>[-<spec>]-c<N>` |
| `date` | `YYYY-MM-DD` | yes | |
| `model` | string | yes | a `models.toml` id |
| `checkpoint` | string | yes | repo @ revision, byte count, pipeline. Pin the revision — "NVFP4" names a format, not a footprint |
| `hardware` | string | yes | a `hardware.toml` id |
| `nodes` | int | no (1) | machines used |
| `parallelism` | string | if nodes > 1 | `TP=2 over 200G RoCE`, `DP=2 replicas` |
| `engine` | string | yes | `llama.cpp` / `vllm` / `sglang` / … |
| `engine_version` | string | yes | tag, image digest, or build date |
| `quant` | string | yes | `Q8_0`, `Q4_K_M`, `NVFP4`, `NVFP4-RTN-experts`, `APEX-mixed`, `AWQ-int4-g32`, `MXFP4`; append ` + fp8 KV` when the KV cache is quantized |
| `checkpoint_gb` | float | no | on-disk size, decimal GB |
| `active_gb_per_token` | float | no | see above |
| `active_gb_method` | `arch` / `fraction` / `assumed` | with the above | |
| `speculation` | `none` / `draft-mtp` / `nextn` / `mtp` / `dflash` / `unknown` | yes | state it: it moves tok/s by 1.6–2× |
| `concurrency` | int | yes | simultaneous streams |
| `decode_tok_s` | float | yes | per-stream median at concurrency 1 (first token excluded); **aggregate** above |
| `decode_per_stream_tok_s` | float | only if concurrency > 1 | |
| `context_tokens` | int | no | prompt length at decode time, when the cell is about long context |
| `prefill_tok_s` | float | no | with `prefill_prompt_tokens` — the **actual** token count, not the nominal label |
| `prefill_prompt_tokens` | int | with the above | |
| `config` | string | no | the flags that mattered: memory fraction, ctx, max running requests, KV dtype, kernel |
| `source` | string | yes | where the raw data lives. A URL, or `lab:<entry>` for a private notebook |
| `notes` | string | no | thinking on/off, acceptance rate, what the number means |

Unknown fields are rejected, so typos fail loudly.

## What makes a row worth submitting

- Single-stream with speculation **off** on a model whose active bytes/token you computed — that
  is a roofline data point. Everything else is context.
- Negative results: a model 3× under its roofline is more useful than one on it.
- The config change that moved a number, in `config`/`notes`, with the before figure.
