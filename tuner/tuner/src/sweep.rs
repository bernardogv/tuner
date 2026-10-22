//! `tuner sweep`: preflight, measure the plan, decide, and write results / casebook draft / recipe.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::bench::{self, KldMode};
use crate::decide::{self, Candidate};
use crate::preflight;
use crate::proc::Captured;
use crate::remote::{self, Remote};
use crate::{gguf, plan};
use tuner::contract::{
    exit, Config, Control, Decision, Depth, Gib, Host, ModelRef, Pct, PlanPoint, Point, Quality, Reason, ReasonKind,
    Recipe, Results, Roofline, Sample, Series, SeriesStats, Spec, Stack, Thresholds, Verdict, Workload, SCHEMA_VERSION,
};

/// Per depth: one depth's pp and tg rows x repeats, sized on depth 0 and fine at 32k (~10 min).
const BENCH_TIMEOUT_PER_DEPTH: Duration = Duration::from_secs(20 * 60);
const KLD_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub struct Args {
    pub host: Host,
    pub model: PathBuf,
    pub spec: PathBuf,
    pub recipes: PathBuf,
    pub out: PathBuf,
    pub reserve: Gib,
    pub spark_mode_ack: Option<String>,
}

/// An exit code plus why, so `main` prints one line and exits.
pub struct Exit(pub i32, pub String);

fn refuse(msg: String) -> Exit {
    Exit(exit::REFUSED, format!("refused: {msg}"))
}

fn failed(msg: impl Into<String>) -> Exit {
    Exit(exit::FAILED, msg.into())
}

/// A KLD step the sweep cannot continue without (baseline logits, control) failed: exit `FAILED`
/// naming the step, why (`kld: ...` from `bench::kld_outcome`, or the run's exit), and the kept log.
pub fn kld_fatal(step: &str, why: &str, log: &Path) -> Exit {
    failed(format!("{step} KLD run failed ({why}); log: {}", log.display()))
}

/// `<out>/<name>.log` <- the run's combined output, the moment the run returns.
fn keep_log(out: &Path, name: &str, run: &Captured) -> Result<PathBuf, Exit> {
    let path = out.join(format!("{name}.log"));
    std::fs::write(&path, run.combined()).map_err(|e| failed(format!("{}: {e}", path.display())))?;
    Ok(path)
}

fn utc(fmt: &str) -> String {
    std::process::Command::new("date")
        .args(["-u", fmt])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

pub fn run(a: &Args) -> Result<(), Exit> {
    let spec = plan::read_spec(&a.spec).map_err(failed)?;
    let model = a.model.to_string_lossy().into_owned();
    if a.host.is_spark() {
        return spark(a, &model);
    }
    server_only_axes(&spec).map_err(failed)?;
    let plan = plan::build(&spec, a.host, &a.model, Some(&a.recipes)).map_err(failed)?;

    let r = Remote::connect(&a.host.to_string(), &utc("+%Y%m%dT%H%M%SZ")).map_err(failed)?;
    let pf = r.ssh(&preflight::script(&model), Duration::from_secs(60)).map_err(failed)?;
    let pf = preflight::parse(&pf.stdout).map_err(failed)?;
    // Model presence, residency and memory first: the GGUF header can only be read once the model
    // is known to exist. Then the same rules again with the logits-file size, which adds the disk rule.
    if let Some(why) = pf.refusal(&model, a.reserve, 0) {
        return Err(refuse(why));
    }
    let model_bytes = pf.model_bytes.unwrap_or(0);
    let active = active(&r, &model, model_bytes).map_err(failed)?;
    let logits = bench::kld_base_bytes(active.n_vocab.unwrap_or(tuner::contract::FALLBACK_N_VOCAB), spec.workload.kld_ctx);
    if let Some(why) = pf.refusal(&model, a.reserve, logits) {
        return Err(refuse(why));
    }
    std::fs::create_dir_all(&a.out).map_err(|e| failed(format!("{}: {e}", a.out.display())))?;

    let reaper = r.clone();
    // A second ctrl-c while reaping is ignored; the handler exits once the groups are gone.
    let _ = ctrlc::set_handler(move || {
        let _ = reaper.reap();
        std::process::exit(130);
    });
    let started_utc = utc("+%Y-%m-%dT%H:%M:%SZ");
    let measured = measure(&r, &a.out, &model, &active, &plan.points, spec.repeats, &spec.workload);
    let reaped = r.reap();
    let (stack, roofline, control, points) = measured?;
    reaped.map_err(|e| failed(format!("reap failed, check {} for strays: {e}", r.host)))?;

    let mut points = points.into_iter();
    let Some(baseline) = points.next() else {
        return Err(failed("plan had no baseline"));
    };
    let results = Results {
        schema_version: SCHEMA_VERSION,
        host: a.host,
        model: ModelRef {
            path: a.model.clone(),
            bytes: model_bytes,
        },
        stack,
        roofline,
        started_utc,
        baseline,
        control,
        points: points.collect(),
    };
    write_outputs(a, &results, &spec.workload)
}

/// llama-bench has no `-c`/`-np`; sweeping them would measure the same thing twice.
fn server_only_axes(spec: &Spec) -> Result<(), String> {
    let varies = |v: &Option<Vec<u32>>, base: u32| v.as_ref().is_some_and(|xs| xs.iter().any(|x| *x != base));
    if varies(&spec.grid.ctx, spec.baseline.ctx) || varies(&spec.grid.np, spec.baseline.np) {
        return Err("ctx / np are server knobs; v0 measures with llama-bench, so they cannot be swept yet".into());
    }
    Ok(())
}

/// GB10 unified memory: refuse unless the operator put the box in an idle mode and says which.
fn spark(a: &Args, model: &str) -> Result<(), Exit> {
    let host = a.host.to_string();
    let Some(ack) = &a.spark_mode_ack else {
        return Err(refuse(format!(
            "{host} is a Spark: pass --spark-mode-ack <mode> after `spark-mode switch <mode>` to a mode that serves nothing"
        )));
    };
    let status = remote::ssh_once(&host, "~/bin/spark-mode status").map_err(failed)?;
    if let Some(why) = preflight::spark_refusal(&status.stdout, ack) {
        return Err(refuse(why));
    }
    let pf = remote::ssh_once(&host, &preflight::script(model)).map_err(failed)?;
    let pf = preflight::parse(&pf.stdout).map_err(failed)?;
    if let Some(why) = pf.refusal(model, a.reserve, 0) {
        return Err(refuse(why));
    }
    Err(Exit(exit::UNIMPLEMENTED, "unimplemented: Spark benchmarking is not in v0".into()))
}

type Measured = (Stack, Roofline, Control, Vec<Point>);

/// Baseline bench, baseline logits, control KLD, then every candidate (contract.rs "Quality" for
/// the order and what each KLD failure means). `out` receives one `kld-*.log` per KLD run.
fn measure(r: &Remote, out: &Path, model: &str, active: &gguf::Active, points: &[PlanPoint], repeats: u32, w: &Workload) -> Result<Measured, Exit> {
    let Some(base_cfg) = points.first().map(|p| p.config) else {
        return Err(failed("empty plan"));
    };
    let stack = stack(r, &base_cfg).map_err(failed)?;
    let roofline = roofline(active).map_err(failed)?;
    let t = Thresholds::default();

    let (base_samples, base_failure) = bench_point(r, "bench-0", model, &base_cfg, repeats, &w.depths)?;
    if let Some(f) = base_failure {
        return Err(failed(format!("baseline crashed, nothing to compare against: {f}")));
    }
    let logits = format!("{}/base.kld", r.run_dir);
    let base_run = r
        .tracked("kld-0", &bench::kld_script(&r.home, model, &base_cfg, &logits, KldMode::Base, w.kld_ctx), KLD_TIMEOUT)
        .map_err(failed)?;
    let base_log = keep_log(out, "kld-0", &base_run)?;
    if !base_run.ok() {
        let why = if base_run.timed_out { "timed out".to_string() } else { format!("exit {:?}", base_run.status) };
        return Err(kld_fatal("baseline logits", &why, &base_log));
    }
    let ctl_cfg = bench::control_config(base_cfg).ok_or_else(|| failed("baseline has no reorder-only control"))?;
    let ctl_run = r
        .tracked("kld-control", &bench::kld_script(&r.home, model, &ctl_cfg, &logits, KldMode::Compare, w.kld_ctx), KLD_TIMEOUT)
        .map_err(failed)?;
    let ctl_log = keep_log(out, "kld-control", &ctl_run)?;
    let control = Control {
        config: ctl_cfg,
        kld: bench::kld_outcome(&ctl_run).map_err(|why| kld_fatal("control", &why, &ctl_log))?,
    };
    eprintln!("control {:?}: KLD {:.4}, same-top {:.1}%", ctl_cfg, control.kld.mean_kld.0, control.kld.same_top_pct.0);

    let decision = Decision {
        verdict: Verdict::Baseline,
        reason: Reason {
            kind: ReasonKind::Baseline,
            detail: "the anchor every candidate is compared against".into(),
        },
    };
    let mut measured = vec![point(base_cfg, repeats, base_samples.clone(), None, None, decision, &roofline, w)];
    for (i, p) in points.iter().enumerate().skip(1) {
        eprintln!("[{i}/{}] {:?}", points.len() - 1, p.config);
        let (samples, mut failure) = bench_point(r, &format!("bench-{i}"), model, &p.config, repeats, &w.depths)?;
        let kld = if failure.is_none() {
            let script = bench::kld_script(&r.home, model, &p.config, &logits, KldMode::Compare, w.kld_ctx);
            let run = r.tracked(&format!("kld-{i}"), &script, KLD_TIMEOUT).map_err(failed)?;
            keep_log(out, &format!("kld-{i}"), &run)?;
            match bench::kld_outcome(&run) {
                Ok(q) => Some(q),
                Err(why) => {
                    failure = Some(why);
                    None
                }
            }
        } else {
            None
        };
        let cand = Candidate {
            samples: &samples,
            kld,
            failure: failure.as_deref(),
            identical_to_baseline: p.config == base_cfg,
        };
        let d = decide::decide(&t, w, repeats, &base_samples, control.kld, &cand);
        eprintln!("  -> {:?}: {}", d.verdict, d.reason.detail);
        measured.push(point(p.config, repeats, samples, failure, kld, d, &roofline, w));
    }
    Ok((stack, roofline, control, measured))
}

fn last_line(s: &str) -> &str {
    s.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim()
}

/// Samples, or the failure that cut the repeats short. Only ssh trouble is an `Err`.
fn bench_point(
    r: &Remote,
    tag: &str,
    model: &str,
    c: &Config,
    repeats: u32,
    depths: &[Depth],
) -> Result<(Vec<Sample>, Option<String>), Exit> {
    let timeout = BENCH_TIMEOUT_PER_DEPTH * u32::try_from(depths.len()).unwrap_or(u32::MAX);
    let script = bench::bench_script(&r.home, model, c, repeats, depths);
    let out = r.tracked(tag, &script, timeout).map_err(failed)?;
    if !out.ok() {
        let why = if out.timed_out { "timed out".to_string() } else { format!("exit {:?}", out.status) };
        return Ok((Vec::new(), Some(format!("{why}: {}", last_line(&out.stderr)))));
    }
    match bench::parse_bench(&out.stdout, depths) {
        Ok(s) => Ok((s, None)),
        Err(e) => Ok((Vec::new(), Some(e))),
    }
}

#[allow(clippy::too_many_arguments)]
fn point(
    config: Config,
    repeats: u32,
    samples: Vec<Sample>,
    failure: Option<String>,
    kld: Option<Quality>,
    decision: Decision,
    roofline: &Roofline,
    w: &Workload,
) -> Point {
    let series: Vec<SeriesStats> = w
        .series()
        .into_iter()
        .filter_map(|s| decide::stats(&samples, s).map(|stats| SeriesStats { series: s, stats }))
        .collect();
    let tg0 = series.iter().find(|s| s.series == Series::TG0).map(|s| s.stats.mean.0);
    Point {
        config,
        repeats,
        roofline_pct: tg0.map(|m| Pct(m / roofline.expected_tg_tok_s.0 * 100.0)),
        series,
        samples,
        failure,
        kld,
        decision,
    }
}

fn stack(r: &Remote, base: &Config) -> Result<Stack, String> {
    let out = r.ssh(
        r#"cd ~/src/llama.cpp && echo "llama=$(git describe --tags 2>/dev/null) $(git rev-parse --short=12 HEAD)"
echo "kernel=$(uname -r)"; echo "profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
echo "vulkan=mesa $(dpkg-query -W -f='${Version}' mesa-vulkan-drivers 2>/dev/null)"
echo "hip=rocm $(dpkg-query -W -f='${Version}' amdrocm-core-sdk10.1-gfx1151 2>/dev/null)""#,
        Duration::from_secs(60),
    )?;
    let get = |k: &str| {
        out.stdout
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{k}=")))
            .unwrap_or("unknown")
            .trim()
            .to_string()
    };
    let driver = match base.backend {
        tuner::contract::Backend::Vulkan => get("vulkan"),
        tuner::contract::Backend::Hip => get("hip"),
    };
    Ok(Stack {
        llama_cpp: get("llama"),
        kernel: get("kernel"),
        platform_profile: get("profile"),
        driver,
    })
}

/// The GGUF header's tensor table and vocab, fetching more of the file if 32 MiB does not cover it.
fn active(r: &Remote, model: &str, model_bytes: u64) -> Result<gguf::Active, String> {
    for n in [32usize << 20, 256 << 20] {
        match gguf::active(&r.head(model, n)?, model_bytes) {
            Ok(a) => return Ok(a),
            Err(gguf::GgufError::Truncated) => continue,
            Err(gguf::GgufError::Invalid(e)) => return Err(format!("{model}: {e}")),
        }
    }
    Err("GGUF header larger than 256 MiB".into())
}

fn roofline(active: &gguf::Active) -> Result<Roofline, String> {
    let active_gb = active.gb_per_token();
    let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benchmarks/data");
    let ds = roofline_benchmarks::Dataset::load(&data)?;
    let id = "strix";
    let hw = ds.hardware(id).ok_or_else(|| format!("no hardware {id:?} in {}", data.display()))?;
    Ok(Roofline {
        hardware_id: id.into(),
        decode_bw_gbs: hw.decode_bw_gbs,
        active_gb_per_token: active_gb,
        expected_tg_tok_s: tuner::contract::TokS(roofline::roofline_tok_s(hw, 1, active_gb)),
    })
}

fn write_outputs(a: &Args, res: &Results, w: &Workload) -> Result<(), Exit> {
    let io = |e: std::io::Error| failed(e.to_string());
    std::fs::create_dir_all(&a.out).map_err(io)?;
    let json = serde_json::to_string_pretty(res).map_err(|e| failed(e.to_string()))?;
    std::fs::write(a.out.join("results.json"), json).map_err(io)?;
    std::fs::write(a.out.join("casebook-entry.md"), casebook(res, w)).map_err(io)?;

    let objective = |p: &Point| mean_of(p, w.objective);
    let winner = res
        .points
        .iter()
        .filter(|p| p.decision.verdict == Verdict::Keep)
        .chain(std::iter::once(&res.baseline))
        .max_by(|x, y| objective(x).total_cmp(&objective(y)))
        .unwrap_or(&res.baseline);
    let recipe = Recipe {
        schema_version: SCHEMA_VERSION,
        host: res.host,
        model: plan::model_name(&a.model),
        winner: winner.config,
        objective: w.objective,
        series: winner.series.clone(),
        recorded_utc: utc("+%Y-%m-%dT%H:%M:%SZ"),
    };
    let path = plan::recipe_path(&a.recipes, res.host, &a.model);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(io)?;
    }
    let json = serde_json::to_string_pretty(&recipe).map_err(|e| failed(e.to_string()))?;
    std::fs::write(&path, json).map_err(io)?;
    eprintln!("results: {}\nrecipe: {}", a.out.join("results.json").display(), path.display());
    Ok(())
}

fn mean_of(p: &Point, s: Series) -> f64 {
    p.series.iter().find(|x| x.series == s).map_or(0.0, |x| x.stats.mean.0)
}

/// Draft entry; one `<series> tok/s` column per series of the baseline (v1: every metric x depth),
/// the KLD protocol (`-c <kld_ctx> --chunks <n>`) and the control's reading (v1.1).
fn casebook(res: &Results, w: &Workload) -> String {
    let name = plan::model_name(&res.model.path);
    let rf = &res.roofline;
    let columns: Vec<Series> = res.baseline.series.iter().map(|x| x.series).collect();
    let header: String = columns.iter().map(|c| format!(" {c} tok/s |")).collect();
    let mut s = format!(
        "# {name}: llama.cpp config sweep on {}, {}\n\nDraft from `roofline-labs/tuner sweep`; review before filing.\n\n## Stack\n\
         - kernel {} · platform_profile {} · llama.cpp {} · driver {}\n\
         - model {name} ({:.1} GB) · llama-bench `-ngl 99 -p 512 -n 128 -r {}`, flags per row (`-fa`, `-ub`, `-b`, `-ctk/-ctv`), `-d` per column\n\
         - KLD: llama-perplexity, wikitext-2 test, `-c {} --chunks {}`, against the baseline's logits; \
           control (baseline with `-ub {}`): KLD {:.4}, same-top {:.1}%; gate is relative to the control\n\
         - roofline: {:.0} GB/s decode (hardware `{}`) / {:.3} GB/token active = {:.2} tok/s expected (tg@0)\n\n## Result\n\n\
         | role | backend | -fa | -ub | -b | -ctk/-ctv | hipBLASLt |{header} KLD | same-top | % roofline | decision |\n\
         |---|---|---|---|---|---|---|{}---|---|---|---|\n",
        res.host,
        res.started_utc.get(..10).unwrap_or(&res.started_utc),
        res.stack.kernel,
        res.stack.platform_profile,
        res.stack.llama_cpp,
        res.stack.driver,
        res.model.bytes as f64 / 1e9,
        res.baseline.repeats,
        w.kld_ctx,
        bench::kld_chunks(w.kld_ctx),
        res.control.config.ub,
        res.control.kld.mean_kld.0,
        res.control.kld.same_top_pct.0,
        rf.decode_bw_gbs,
        rf.hardware_id,
        rf.active_gb_per_token,
        rf.expected_tg_tok_s.0,
        "---|".repeat(columns.len()),
    );
    for (role, p) in std::iter::once(("baseline", &res.baseline)).chain(res.points.iter().map(|p| ("candidate", p))) {
        let c = &p.config;
        let cells: String = columns
            .iter()
            .map(|col| {
                p.series
                    .iter()
                    .find(|x| x.series == *col)
                    .map_or(" – |".into(), |x| format!(" {:.2} ± {:.2} |", x.stats.mean.0, x.stats.stdev.0))
            })
            .collect();
        let (kld, top) = p.kld.map_or(("–".into(), "–".into()), |q| (format!("{:.4}", q.mean_kld.0), format!("{:.1}%", q.same_top_pct.0)));
        let _ = writeln!(
            s,
            "| {role} | {:?} | {} | {} | {} | {:?}/{:?} | {} |{cells} {kld} | {top} | {} | {:?}: {} |",
            c.backend,
            c.fa,
            c.ub,
            c.b,
            c.ctk,
            c.ctv,
            c.hipblaslt,
            p.roofline_pct.map_or("–".into(), |x| format!("{:.0}%", x.0)),
            p.decision.verdict,
            p.decision.reason.detail
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline-logits and control failures end the sweep (`FAILED`), pointing at the kept log;
    /// they are never dressed up as a quality verdict on the candidates.
    #[test]
    fn a_failed_instrument_step_fails_the_sweep_naming_its_log() {
        let log = Path::new("/tmp/out/kld-0.log");
        let Exit(code, msg) = kld_fatal("baseline logits", "exit 1", log);
        assert_eq!(code, exit::FAILED);
        assert!(msg.contains("baseline logits") && msg.contains("exit 1") && msg.contains("/tmp/out/kld-0.log"), "{msg}");
        let Exit(code, msg) = kld_fatal("control", "kld: no Mean KLD / Same top p in output: PPL = 1095", Path::new("kld-control.log"));
        assert_eq!(code, exit::FAILED);
        assert!(msg.contains("control") && msg.contains("kld-control.log"), "{msg}");
    }
}
