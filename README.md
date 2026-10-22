# tuner

A deterministic llama.cpp config tuner for a single box, scored against the memory-bandwidth
roofline. It sweeps llama.cpp knobs (backend, `-ub`, KV cache types, flash attention, hipBLASLt),
benchmarks every point at the context depths you care about, and keeps a config only if it is
faster on the metric you named **without** losing quality or regressing anything else.

Built for an AMD Strix Halo box (Ryzen AI Max+ 395, Radeon 8060S, gfx1151, 128 GB unified memory);
the DGX Spark (GB10) path refuses by default and is not implemented yet.

## Layout

| dir | what |
|---|---|
| `tuner/` | the tuner (`tuner plan / sweep / decide`), its acceptance harness (`accept`), and `longctx`, a multi-needle long-context retrieval check |
| `cli/` | `roofline`: hardware + model shape + quant → expected decode tok/s and memory fit |
| `benchmarks/` | measured decode speeds scored against the roofline; the hardware rows the tuner reads |

## The keep rule

Every candidate is benchmarked (`llama-bench -p 512 -n 128 -d <depths> -r <repeats>`) and its
KL divergence measured against the baseline's logits (`llama-perplexity`). Then, in order:

1. **Crashed**: a run failed, a depth is short of repeats, or there is no KLD reading.
2. **IdenticalConfig**, **TooFewSamples**.
3. **QualityGate**, relative to a *reorder-only control* (the baseline with only `-ub` changed):
   KLD ≤ control + 0.02 and same-top ≥ control − 3 pts. Any kernel change drifts KLD (≈0.03 at
   512 ctx, ≈0.08 at 32k on gpt-oss-120b), so an absolute threshold would reject harmless changes.
4. **GuardRegression**: no non-objective series (pp/tg at each depth) may lose more than
   `guard_pct` (default 3%).
5. **NoiseGate**: the objective must beat the baseline by a Welch lower bound > 0 (k = 2).
6. **Cleared** → keep. The best kept point becomes the recipe and warm-starts the next sweep.

A spec says what matters:

```toml
schema_version = 3
repeats = 3

[workload]
depths = [0, 32768]
objective = { metric = "pp", depth = 32768 }   # prefill at 32k must improve
guard_pct = 3.0                                # nothing else may lose > 3%
kld_ctx = 512                                  # or 32768 to judge quality at depth (~13 GB logits file)

[baseline]
backend = "vulkan"
fa = true
ub = 512
b = 2048
ctk = "f16"
ctv = "f16"
ctx = 8192
np = 1
hipblaslt = false

[grid]
ub = [512, 1024, 2048]
ctk = ["f16", "q8_0"]
ctv = ["f16", "q8_0"]
```

## Example result (Strix Halo, gpt-oss-120b MXFP4, Vulkan, 2026-10)

| config | prefill @32k | decode @0 | decode @32k | decision |
|---|---|---|---|---|
| f16/f16 KV (baseline) | 369 tok/s | 54.5 | 41.3 | — (≈100% of the decode roofline) |
| f16 K / **q8_0 V** | 472 (+28%) | −1.9% | +4.6% | keep |
| **q8_0/q8_0** KV | 565 (+53%) | −1.7% | +4.9% | keep (recipe winner) |
| HIP + hipBLASLt | ~510 | −7% | −7% | reject at the 3% guard |

q8_0 V was then checked at 32k (KLD +0.007 over the control) and on a 24-prompt long-context
retrieval task (47/48 for both q8_0 V and f16), and deployed.

## Running

```sh
cd tuner
cargo run -p tuner -- plan  --host strix --model /path/on/host/model.gguf --spec sweep.toml
cargo run -p tuner -- sweep --host strix --model /path/on/host/model.gguf --spec sweep.toml --out out/
cargo run -p accept                   # offline acceptance checks
TUNER_GGUF=/path/on/host/model.gguf cargo run -p accept -- --live [--kld-32k]
```

The host is an ssh alias. It needs llama.cpp builds at `~/src/llama.cpp/build-{vulkan,hip}/bin`
and wikitext-2 at `~/datasets/wikitext-2-raw/`.

## Safety

The tuner runs on a machine that also serves models, so it refuses before touching anything if:
a model is resident in llama-swap (`:8089`) or any llama.cpp process is running; memory headroom
after the model is below `--reserve-gib`; or free disk can't hold the KLD logits file. Every
launch runs in its own process group and is reaped on exit or ctrl-c, and logits files are deleted.

## License

Code: MIT ([LICENSE](LICENSE)). Benchmark data in `benchmarks/data/`: CC-BY-4.0.
