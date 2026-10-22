//! `tuner` CLI. The argument surface is the contract (see `contract.rs`).

mod bench;
mod decide;
mod gguf;
mod plan;
mod preflight;
mod proc;
mod remote;
mod sweep;

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use tuner::contract::{exit, DecideFixture, DecideOutput, Gib, Host, DEFAULT_RESERVE_GIB, SCHEMA_VERSION};

#[derive(Parser)]
#[command(name = "tuner", version, about = "Deterministic llama.cpp config tuner, scored against the roofline")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args)]
struct Target {
    #[arg(long, value_enum)]
    host: Host,
    /// GGUF path on the host.
    #[arg(long)]
    model: PathBuf,
    /// sweep.toml (`contract::Spec`).
    #[arg(long)]
    spec: PathBuf,
    /// Recipe store; `<dir>/<host>/<model stem>.json` warm-starts the plan.
    #[arg(long)]
    recipes: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print the expanded plan as JSON. No side effects.
    Plan {
        #[command(flatten)]
        target: Target,
    },
    /// Run the plan on the host; write results.json, casebook-entry.md, and the recipe.
    Sweep {
        #[command(flatten)]
        target: Target,
        #[arg(long)]
        out: PathBuf,
        /// Refuse unless MemAvailable - model size >= this many GiB.
        #[arg(long, default_value_t = DEFAULT_RESERVE_GIB)]
        reserve_gib: f64,
        /// Spark hosts only: the mode you put the box in with `spark-mode switch <mode>`.
        /// Without it a Spark sweep is refused before anything is contacted.
        #[arg(long)]
        spark_mode_ack: Option<String>,
    },
    /// Apply the keep rule to a `contract::DecideFixture` and print a `contract::DecideOutput`.
    Decide {
        #[arg(long)]
        fixture: PathBuf,
    },
}

fn code(c: i32) -> ExitCode {
    ExitCode::from(u8::try_from(c).unwrap_or(1))
}

fn fail(msg: &str) -> ExitCode {
    eprintln!("error: {msg}");
    code(exit::FAILED)
}

fn print_json<T: serde::Serialize>(v: &T) -> ExitCode {
    match serde_json::to_string_pretty(v) {
        Ok(text) => {
            println!("{text}");
            code(exit::OK)
        }
        Err(e) => fail(&e.to_string()),
    }
}

fn run_plan(t: &Target) -> ExitCode {
    let built = plan::read_spec(&t.spec).and_then(|spec| plan::build(&spec, t.host, &t.model, t.recipes.as_deref()));
    match built {
        Ok(p) => print_json(&p),
        Err(e) => fail(&e),
    }
}

fn run_decide(fixture: &Path) -> ExitCode {
    let parsed = std::fs::read_to_string(fixture)
        .map_err(|e| e.to_string())
        .and_then(|text| serde_json::from_str::<DecideFixture>(&text).map_err(|e| e.to_string()));
    let f = match parsed {
        Ok(f) => f,
        Err(e) => return fail(&format!("{}: {e}", fixture.display())),
    };
    let cand = decide::Candidate {
        samples: &f.candidate.samples,
        kld: f.candidate.kld,
        failure: f.candidate.failure.as_deref(),
        identical_to_baseline: f.candidate.identical_to_baseline,
    };
    print_json(&DecideOutput {
        schema_version: SCHEMA_VERSION,
        decision: decide::decide(&f.thresholds, &f.workload, f.repeats, &f.baseline.samples, f.control, &cand),
        deltas: decide::deltas(&f.workload, &f.baseline.samples, &f.candidate.samples),
    })
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Plan { target } => run_plan(&target),
        Cmd::Sweep {
            target,
            out,
            reserve_gib,
            spark_mode_ack,
        } => {
            let args = sweep::Args {
                host: target.host,
                model: target.model,
                spec: target.spec,
                // The repo-level store, so successive runs warm-start without a flag.
                recipes: target
                    .recipes
                    .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../recipes")),
                out,
                reserve: Gib(reserve_gib),
                spark_mode_ack,
            };
            match sweep::run(&args) {
                Ok(()) => code(exit::OK),
                Err(sweep::Exit(c, msg)) => {
                    eprintln!("{msg}");
                    code(c)
                }
            }
        }
        Cmd::Decide { fixture } => run_decide(&fixture),
    }
}
