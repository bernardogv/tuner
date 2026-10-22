//! `accept`: the acceptance harness for `tuner`.
//! `cargo run -p accept -- [--only A3 ...] [--json] [--live]`; exit 1 iff any check fails.
//! Report: accept/out/accept-<run id>.json.

mod checks;
mod proc;
mod remote;
mod rules;
mod status;

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;

use checks::Ctx;
use status::{exit_code, summary, tally, RunReport};

#[derive(Parser)]
#[command(name = "accept", about = "Acceptance checks for tuner")]
struct Args {
    /// Check ids to run (A1..A8, A4a, A4b); default all.
    #[arg(long)]
    only: Vec<String>,
    /// Print the JSON report to stdout instead of the text summary.
    #[arg(long)]
    json: bool,
    /// Allow A5 and A7, the real sweeps on strix (still gated by the harness preflight).
    #[arg(long)]
    live: bool,
    /// With --live, also allow A8: KLD at 32k context on strix (~20 min, ~13 GB of disk there).
    #[arg(long)]
    kld_32k: bool,
}

/// RFC 3339 UTC from the system clock, no chrono (Howard Hinnant's civil-from-days).
pub fn utc_now() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn main() -> ExitCode {
    let args = Args::parse();
    let accept_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = accept_dir.parent().map(PathBuf::from).unwrap_or_else(|| accept_dir.clone());
    let started = utc_now();
    let run_id = started.replace(['-', ':'], "").trim_end_matches('Z').to_string();
    let ctx = Ctx {
        only: args.only,
        live: args.live,
        kld_32k: args.kld_32k,
        tuner_bin: root.join("target/debug/tuner"),
        fixtures: accept_dir.join("fixtures"),
        out: accept_dir.join("out"),
        root,
        run_id,
    };
    let _ = fs::create_dir_all(&ctx.out);
    let checks = checks::run_selected(&ctx);
    let (passed, failed, skipped) = tally(&checks);
    let code = exit_code(&checks);
    let report = RunReport { run_id: ctx.run_id.clone(), started_utc: started, live: ctx.live, checks, passed, failed, skipped };
    let json = serde_json::to_string_pretty(&report).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"));
    let path = ctx.out.join(format!("accept-{}.json", ctx.run_id));
    if let Err(e) = fs::write(&path, &json) {
        eprintln!("cannot write {}: {e}", path.display());
    }
    if args.json {
        println!("{json}");
    } else {
        print!("{}", summary(&report));
        println!("report: {}", path.display());
    }
    ExitCode::from(u8::try_from(code).unwrap_or(1))
}

#[cfg(test)]
mod tests {
    #[test]
    fn utc_now_is_rfc3339_shaped() {
        let s = super::utc_now();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.starts_with("20") && s.ends_with('Z') && &s[10..11] == "T", "{s}");
    }
}
