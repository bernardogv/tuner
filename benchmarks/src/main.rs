use roofline_benchmarks::{Dataset, html, render};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: roofline-benchmarks <validate|render|render-html> [data-dir]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let dir = PathBuf::from(args.get(1).map_or("data", String::as_str));
    let ds = match Dataset::load(&dir) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("load failed: {e}");
            return ExitCode::from(1);
        }
    };
    let errs = ds.validate();
    if !errs.is_empty() {
        for e in &errs {
            eprintln!("{e}");
        }
        eprintln!("{} problem(s)", errs.len());
        return ExitCode::from(1);
    }
    match cmd.as_str() {
        "validate" => {
            eprintln!(
                "ok: {} hardware, {} models, {} runs",
                ds.hardware.len(),
                ds.models.len(),
                ds.runs.len()
            );
            ExitCode::SUCCESS
        }
        "render" => {
            print!("{}", render::markdown(&ds));
            ExitCode::SUCCESS
        }
        "render-html" => {
            print!("{}", html::page(&ds));
            ExitCode::SUCCESS
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}
