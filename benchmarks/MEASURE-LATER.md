# Measure later

Gaps in the table that need hardware time. Nothing here is a guess in `data/` — rows are
either measured or absent.

## Missing cells (hardware idle required)

- **Nemotron 3.5 Lightning, speculation off, single stream.** The production resident has only
  ever been measured with NEXTN on; its 96% of roofline is inferred by dividing by ~1.2. One
  `decode-c1` cell with `--speculative-algorithm` unset settles it.
- **Qwen3-Coder-Next NVFP4 at 32% of roofline — why.** Third-party llm-compressor requant on
  SGLang. Cells to run: same checkpoint on vLLM (engine or checkpoint?); `Q4_K_M` GGUF on
  llama.cpp (kernel path?); a ModelOpt NVFP4 of the same base if one appears.
- **A no-speculation cell for every `draft-mtp` llama.cpp row** (heretic Q8_0, heretic
  NVFP4-RTN, APEX). The APEX "dequant-compute-bound" diagnosis rests on one spec-on number.
- **llama.cpp current build on the 2026-05 models** (Nemotron-3-Nano Q8_0 38.8, gpt-oss-120b
  MXFP4 40.6). Those were an early build; the 35B-A3B class moved 38.8 → 55–58 by August.
- **SGLang with bf16 KV.** Every SGLang row is fp8 KV (engine default); the quality/throughput
  effect of its own default was never isolated.
- **NVFP4 MoE with a non-Marlin kernel on GB10.** Every vLLM row is the Marlin weight-only path.
  Re-test when FlashInfer ships sm_121 NVFP4 MoE kernels; the vLLM half of the engine comparison
  would change.
- **NGC vLLM on a CUDA-13.0-matched tag.** The 26.08 tag failed on CUDA forward-compat, not on
  the engine. Untried, not ruled out.
- **Mistral Small 4 on any engine other than vLLM** — SGLang dies on a triton MLA shape mismatch.
- **Prefill vs prompt length as a first-class table.** Prefill runs at ~4% of BF16 peak on every
  engine; today each c1 row carries one 32k figure and the rest sit in notes.

## Missing derivations (no hardware needed)

- `active_gb_per_token` for **Mistral Small 4 119B** (need the active-param count and which
  tensors are NVFP4 vs BF16; the checkpoint's `quantization_config.ignore` list has it).
- `active_gb_per_token` for **gpt-oss-120b MXFP4** and **Nemotron-3-Nano Q8_0** from their
  layer shapes.
- Exact (`arch`) figures for the four rows that use the `fraction` approximation
  (Nemotron 3.5 2.01, Qwen3-Coder-Next 1.66, Laguna-S 4.54 / 6.10). Attention is Q8/BF16 in all
  of them, so the true number is somewhat higher and the true % somewhat lower.

## Hardware profiles wanted

Only `gb10` exists. The schema is ready for: a second GB10 owner's numbers (reproduction),
RTX 6000 Blackwell / 5090 (discrete VRAM — `memory_kind` differs), Apple M-series (unified,
different STREAM fraction), MI300X / MI355X. Each needs its own `decode_bw_gbs` anchor: one
Q8_0 MoE, no speculation, single stream, and a STREAM copy figure.
