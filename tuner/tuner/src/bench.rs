//! llama-bench and llama-perplexity invocations for one `Config`, their output parsers, and the
//! v1.1 quality helpers (contract.rs "Quality"): chunk count and logits size for a `kld_ctx`, the
//! reorder-only control, and what a finished KLD run means.

use crate::proc::{sh_quote, Captured};
use tuner::contract::{Backend, Config, Depth, KvType, Kld, Pct, Quality, Sample, TokS};

/// wikitext-2 test: the fleet's KLD dataset (`strix/scripts/llama-kld.sh`).
const KLD_DATA: &str = "datasets/wikitext-2-raw/wiki.test.raw";

/// llama-perplexity `--chunks` for a `-c kld_ctx`: `max(2, 16384 / kld_ctx)`, so 32 at 512 (the v0
/// protocol, 8k scored tokens) and 2 at 32768 (32k scored). Total for every `kld_ctx` (0 -> 2).
pub fn kld_chunks(kld_ctx: u32) -> u32 {
    (16_384u32.checked_div(kld_ctx).unwrap_or(0)).max(2)
}

/// Size of the baseline logits file: `n_vocab * 2 bytes * scored tokens`, scored tokens being
/// `kld_chunks(kld_ctx) * kld_ctx / 2` (gpt-oss, 201088: 3_294_625_792 at 512, 13_178_503_168 at
/// 32768; the casebook measured 13.2 GB). Called by the sweep's preflight once implemented.
pub fn kld_base_bytes(n_vocab: u64, kld_ctx: u32) -> u64 {
    n_vocab * 2 * (u64::from(kld_chunks(kld_ctx)) * u64::from(kld_ctx) / 2)
}

/// The reorder-only control: `base` with `ub` doubled if that stays `<= b`, else halved. `None`
/// only when neither differs from the baseline (`ub` 1 with `b` 1).
pub fn control_config(base: Config) -> Option<Config> {
    let ub = if base.ub * 2 <= base.b { base.ub * 2 } else { base.ub / 2 };
    (ub != base.ub && ub > 0).then_some(Config { ub, ..base })
}

/// What a finished KLD compare run means: its `Quality`, or the `Point::failure` string
/// `kld: <exit N | timed out | no Mean KLD / Same top p in output>: <last non-empty line>`.
pub fn kld_outcome(run: &Captured) -> Result<Quality, String> {
    // The KLD script folds stderr into stdout (`2>&1`); `combined` would add the harness's own markers.
    let log = if run.stdout.trim().is_empty() { run.stderr.as_str() } else { run.stdout.as_str() };
    let last = log.lines().rev().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let why = if run.timed_out {
        "timed out".to_string()
    } else if run.status != Some(0) {
        match run.status {
            Some(n) => format!("exit {n}"),
            None => "killed by a signal".to_string(),
        }
    } else {
        return parse_kld(log).ok_or_else(|| format!("kld: no Mean KLD / Same top p in output: {last}"));
    };
    Err(format!("kld: {why}: {last}"))
}

fn kv(t: KvType) -> &'static str {
    match t {
        KvType::F16 => "f16",
        KvType::Q8_0 => "q8_0",
        KvType::Q4_0 => "q4_0",
    }
}

fn bin(home: &str, b: Backend) -> String {
    let dir = match b {
        Backend::Vulkan => "build-vulkan",
        Backend::Hip => "build-hip",
    };
    format!("{home}/src/llama.cpp/{dir}/bin")
}

fn env(c: &Config) -> &'static str {
    match (c.backend, c.hipblaslt) {
        (Backend::Hip, true) => "ROCBLAS_USE_HIPBLASLT=1 ",
        _ => "",
    }
}

/// The flags both tools share. `ctx`/`np` are server knobs: llama-bench sizes its own context.
pub fn flags(c: &Config) -> String {
    format!(
        "-ngl 99 -fa {} -ub {} -b {} -ctk {} -ctv {}",
        if c.fa { "on" } else { "off" },
        c.ub,
        c.b,
        kv(c.ctk),
        kv(c.ctv)
    )
}

/// `<env><bin>/llama-bench -m <model> <flags> -p 512 -n 128 -d <depths, comma-joined> -r <repeats> -o json`.
/// `-d` is always passed (`-d 0` for the default workload) so the script states what it measured.
pub fn bench_script(home: &str, model: &str, c: &Config, repeats: u32, depths: &[Depth]) -> String {
    let d: Vec<String> = depths.iter().map(|d| d.0.to_string()).collect();
    format!(
        "{}{}/llama-bench -m {} {} -p 512 -n 128 -d {} -r {repeats} -o json",
        env(c),
        bin(home, c.backend),
        sh_quote(model),
        flags(c),
        d.join(",")
    )
}

#[derive(Debug, Clone, Copy)]
pub enum KldMode {
    /// Save the baseline's logits.
    Base,
    /// Compare against them.
    Compare,
}

/// `<env><bin>/llama-perplexity -m <model> -f <wikitext> -c <kld_ctx> --chunks <kld_chunks> <flags>
/// --kl-divergence-base <logits> [--kl-divergence] 2>&1`.
pub fn kld_script(home: &str, model: &str, c: &Config, logits: &str, mode: KldMode, kld_ctx: u32) -> String {
    let cmp = match mode {
        KldMode::Base => "",
        KldMode::Compare => " --kl-divergence",
    };
    format!(
        "{}{}/llama-perplexity -m {} -f {home}/{KLD_DATA} -c {kld_ctx} --chunks {} {} --kl-divergence-base {}{cmp} 2>&1",
        env(c),
        bin(home, c.backend),
        sh_quote(model),
        kld_chunks(kld_ctx),
        flags(c),
        sh_quote(logits)
    )
}

/// llama-bench `-o json`: per depth, one pp row (`n_prompt > 0, n_gen == 0`) and one tg row, each
/// with `n_depth` (absent means 0) and per-repeat `samples_ts`. Returns one `Sample` per
/// (repeat, depth), depth-major in the order of `depths`, pairing a depth's pp and tg rows by
/// repeat index. Rows at depths not asked for are ignored; a depth asked for but lacking a pp or
/// tg row is an error.
pub fn parse_bench(json: &str, depths: &[Depth]) -> Result<Vec<Sample>, String> {
    let rows: Vec<serde_json::Value> = serde_json::from_str(json).map_err(|e| format!("llama-bench json: {e}"))?;
    let samples = |depth: Depth, want_pp: bool| -> Option<Vec<f64>> {
        rows.iter()
            .find(|r| {
                let p = r["n_prompt"].as_u64().unwrap_or(0);
                let g = r["n_gen"].as_u64().unwrap_or(0);
                let d = r["n_depth"].as_u64().unwrap_or(0);
                d == u64::from(depth.0) && if want_pp { p > 0 && g == 0 } else { g > 0 && p == 0 }
            })
            .and_then(|r| r["samples_ts"].as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_f64).collect())
    };
    let mut out = Vec::new();
    for &depth in depths {
        let (pp, tg) = samples(depth, true)
            .zip(samples(depth, false))
            .ok_or_else(|| format!("llama-bench json lacks a pp or tg row at depth {}", depth.0))?;
        out.extend(pp.iter().zip(&tg).map(|(p, t)| Sample {
            depth,
            pp_tok_s: TokS(*p),
            tg_tok_s: TokS(*t),
        }));
    }
    Ok(out)
}

/// `Mean    KLD:   0.035443 ±   0.000614` and `Same top p: 87.045 ± 0.296 %`.
pub fn parse_kld(log: &str) -> Option<Quality> {
    let after = |prefix: &str| -> Option<f64> {
        log.lines()
            .find_map(|l| l.trim_start().strip_prefix(prefix))
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|v| v.parse().ok())
    };
    Some(Quality {
        mean_kld: Kld(after("Mean    KLD:")?),
        same_top_pct: Pct(after("Same top p:")?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// strix, 2026-10-05 20:23, gpt-oss-120b MXFP4 on Vulkan, `-d 0,32768 -r 3` (strix-casebook
    /// 2026-10-05-gpt-oss-120b-vulkan-vs-hip, "performance" re-run), trimmed to the keys that matter.
    const TWO_DEPTHS: &str = include_str!("../fixtures/llama-bench-d0-d32768.json");

    fn cfg() -> Config {
        Config {
            backend: Backend::Hip,
            fa: true,
            ub: 512,
            b: 2048,
            ctk: KvType::F16,
            ctv: KvType::Q8_0,
            ctx: 8192,
            np: 1,
            hipblaslt: true,
        }
    }

    #[test]
    fn script_names_every_depth_and_the_right_build() {
        let mut c = cfg();
        let hip = bench_script("/h", "/m.gguf", &c, 3, &[Depth(0)]);
        assert!(hip.starts_with("ROCBLAS_USE_HIPBLASLT=1 /h/src/llama.cpp/build-hip/bin/llama-bench -m /m.gguf "), "{hip}");
        assert!(hip.contains(" -p 512 -n 128 -d 0 -r 3 -o json"), "{hip}");
        c.backend = Backend::Vulkan;
        let vk = bench_script("/h", "/m.gguf", &c, 5, &[Depth(0), Depth(32768)]);
        assert!(vk.starts_with("/h/src/llama.cpp/build-vulkan/bin/llama-bench"), "{vk}");
        assert!(vk.contains(" -p 512 -n 128 -d 0,32768 -r 5 -o json"), "{vk}");
        assert!(flags(&c).contains("-ctv q8_0"));
    }

    #[test]
    fn pairs_pp_and_tg_by_repeat_within_each_depth() {
        let s = parse_bench(TWO_DEPTHS, &[Depth(0), Depth(32768)]).unwrap();
        assert_eq!(s.len(), 6);
        let at = |d: u32| -> Vec<(f64, f64)> {
            s.iter().filter(|x| x.depth == Depth(d)).map(|x| (x.pp_tok_s.0, x.tg_tok_s.0)).collect()
        };
        assert_eq!(at(0), vec![(1085.98, 54.0021), (1088.71, 53.8048), (1087.26, 53.7993)]);
        assert_eq!(at(32768), vec![(355.625, 41.6372), (354.183, 41.6444), (350.924, 41.6602)]);
        assert!(s[..3].iter().all(|x| x.depth == Depth(0)) && s[3..].iter().all(|x| x.depth == Depth(32768)), "depth-major");
        // Fewer depths asked than measured: only those. A depth never measured: an error.
        assert_eq!(parse_bench(TWO_DEPTHS, &[Depth(0)]).unwrap().len(), 3);
        assert!(parse_bench(TWO_DEPTHS, &[Depth(0), Depth(65536)]).is_err());
    }

    #[test]
    fn v0_rows_without_n_depth_are_depth_zero() {
        let j = r#"[{"n_prompt":512,"n_gen":0,"samples_ts":[1000.0,1010.0]},
                    {"n_prompt":0,"n_gen":128,"samples_ts":[54.1,54.3]}]"#;
        let s = parse_bench(j, &[Depth(0)]).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!((s[1].depth, s[1].pp_tok_s.0, s[1].tg_tok_s.0), (Depth(0), 1010.0, 54.3));
        assert!(parse_bench(r#"[{"n_prompt":512,"n_gen":0,"samples_ts":[1.0]}]"#, &[Depth(0)]).is_err(), "no tg row");
    }

    const KLD_LOG: &str = "====== KL divergence statistics ======\nMean    KLD:   0.035443 ±   0.000614\nMaximum KLD:  5.1\n\
                           ====== Token probability statistics ======\nSame top p: 87.045 ± 0.296 %\n";

    #[test]
    fn reads_kld_and_same_top() {
        let q = parse_kld(KLD_LOG).unwrap();
        assert_eq!((q.mean_kld.0, q.same_top_pct.0), (0.035443, 87.045));
        assert_eq!(parse_kld("Final estimate: PPL = 5.0"), None);
    }

    /// v1.1: `-c 512` keeps the v0 flags byte for byte; 32768 is the casebook's `--chunks 2`.
    #[test]
    fn kld_script_follows_kld_ctx() {
        let c = cfg();
        let v0 = kld_script("/h", "/m.gguf", &c, "/h/.tuner/runs/x/base.kld", KldMode::Base, 512);
        assert_eq!(
            v0,
            "ROCBLAS_USE_HIPBLASLT=1 /h/src/llama.cpp/build-hip/bin/llama-perplexity -m /m.gguf -f /h/datasets/wikitext-2-raw/wiki.test.raw \
             -c 512 --chunks 32 -ngl 99 -fa on -ub 512 -b 2048 -ctk f16 -ctv q8_0 --kl-divergence-base /h/.tuner/runs/x/base.kld 2>&1"
        );
        let deep = kld_script("/h", "/m.gguf", &c, "/h/.tuner/runs/x/base.kld", KldMode::Compare, 32768);
        assert!(deep.contains(" -c 32768 --chunks 2 "), "{deep}");
        assert!(deep.ends_with(" --kl-divergence-base /h/.tuner/runs/x/base.kld --kl-divergence 2>&1"), "{deep}");
        assert_eq!((kld_chunks(512), kld_chunks(2048), kld_chunks(8192), kld_chunks(16384), kld_chunks(32768)), (32, 8, 2, 2, 2));
        assert_eq!(kld_chunks(0), 2, "total: a bad kld_ctx fails in llama-perplexity, not here");
    }

    /// gpt-oss (n_vocab 201088): the casebook's 13.2 GB base file at 32k; 3.3 GB at the v0 512.
    #[test]
    fn logits_file_estimate() {
        assert_eq!(kld_base_bytes(201_088, 32768), 13_178_503_168);
        assert_eq!(kld_base_bytes(201_088, 512), 3_294_625_792);
        // kld_ctx 2: max(2, 16384 / 2) = 8192 chunks x 1 scored token each, 2 bytes per token.
        assert_eq!(kld_base_bytes(1, 2), 16_384);
    }

    /// Double while it fits under `b`, else halve; nothing but `ub` changes.
    #[test]
    fn control_differs_from_the_baseline_only_in_ub() {
        let base = Config { backend: Backend::Vulkan, hipblaslt: false, ..cfg() };
        let ctl = control_config(base).unwrap();
        assert_eq!(ctl, Config { ub: 1024, ..base }, "512/2048 doubles");
        assert_eq!(control_config(Config { ub: 1024, ..base }).unwrap().ub, 2048, "1024/2048 doubles to exactly b");
        assert_eq!(control_config(Config { ub: 2048, ..base }).unwrap().ub, 1024, "ub == b halves");
        assert_eq!(control_config(Config { ub: 4096, b: 2048, ..base }).unwrap().ub, 2048, "ub > b halves");
        assert_eq!(control_config(Config { ub: 1, b: 1, ..base }), None, "no other ub exists");
        assert_eq!(control_config(Config { ub: 1, b: 2, ..base }).unwrap().ub, 2);
    }

    /// The sweep-level seam for "a KLD failure is a crash, not a quality verdict": every way the run
    /// can fail yields a `kld:` failure string that names the way and the log's last line.
    #[test]
    fn kld_outcome_is_quality_or_a_named_failure() {
        let run = |status: Option<i32>, timed_out: bool, stdout: &str| Captured {
            status,
            stdout: stdout.into(),
            stderr: String::new(),
            timed_out,
        };
        let q = kld_outcome(&run(Some(0), false, KLD_LOG)).unwrap();
        assert_eq!((q.mean_kld.0, q.same_top_pct.0), (0.035443, 87.045));

        let crashed = kld_outcome(&run(Some(1), false, "llama_model_load: loading model\nggml_vulkan: device lost\n")).unwrap_err();
        assert_eq!(crashed, "kld: exit 1: ggml_vulkan: device lost");
        let hung = kld_outcome(&run(None, true, "perplexity: calculating perplexity over 2 chunks\n\n")).unwrap_err();
        assert_eq!(hung, "kld: timed out: perplexity: calculating perplexity over 2 chunks");
        // The casebook's ub 2048 f16/f16 case: exit 0, nothing parsable. Not a quality result.
        let silent = kld_outcome(&run(Some(0), false, "Final estimate: PPL = 1095.2 +/- 9.1\n")).unwrap_err();
        assert_eq!(silent, "kld: no Mean KLD / Same top p in output: Final estimate: PPL = 1095.2 +/- 9.1");
        assert!(kld_outcome(&run(Some(0), false, "")).unwrap_err().starts_with("kld: no Mean KLD"), "empty output");
    }
}
