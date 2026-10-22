//! Check registry and shared context. Each check module returns `CheckResult`s with evidence.

pub mod a1;
pub mod a2;
pub mod a3;
pub mod a4;
pub mod a5;
pub mod a6;
pub mod a7;
pub mod a8;
pub mod live;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::proc::{self, Captured};
use crate::status::{CheckResult, Evidence, Status};

/// GGUF (path on strix) the strix checks point at: `TUNER_GGUF`, else this placeholder, which the
/// live checks then report as missing on the host. A single-file gpt-oss-120b MXFP4 GGUF.
const DEFAULT_GGUF: &str = "/models/gpt-oss-120b-gguf/gpt-oss-120b-MXFP4.gguf";

pub fn strix_gguf() -> &'static str {
    static PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    PATH.get_or_init(|| std::env::var("TUNER_GGUF").unwrap_or_else(|_| DEFAULT_GGUF.into()))
}

pub struct Ctx {
    pub only: Vec<String>,
    pub live: bool,
    /// A8 (32k-context KLD, ~20 min and ~13 GB on strix) runs only with this and `live`.
    pub kld_32k: bool,
    /// Workspace root (parent of `accept/`).
    pub root: PathBuf,
    pub tuner_bin: PathBuf,
    pub fixtures: PathBuf,
    pub out: PathBuf,
    pub run_id: String,
}

impl Ctx {
    /// `--only A4` selects A4, A4a, A4b; `--only A4a` selects just that.
    pub fn selected(&self, id: &str) -> bool {
        self.only.is_empty() || self.only.iter().any(|o| id.eq_ignore_ascii_case(o) || id.to_lowercase().starts_with(&o.to_lowercase()))
    }

    pub fn fixture(&self, rel: &str) -> PathBuf {
        self.fixtures.join(rel)
    }

    /// Run the tuner binary with `args`, from the workspace root.
    pub fn tuner(&self, args: &[&str], timeout: Duration, grace: Duration) -> Result<Captured, String> {
        let mut cmd = Command::new(&self.tuner_bin);
        cmd.args(args).current_dir(&self.root);
        proc::run(cmd, None, timeout, grace)
    }

    pub fn cargo(&self, args: &[&str], timeout: Duration) -> Result<Captured, String> {
        let mut cmd = Command::new("cargo");
        cmd.args(args).current_dir(&self.root);
        proc::run(cmd, None, timeout, Duration::ZERO)
    }
}

pub fn describe(bin: &Path, args: &[&str]) -> String {
    let name = bin.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    format!("{name} {}", args.join(" "))
}

fn timed(id: &str, name: &str, f: impl FnOnce() -> CheckResult) -> CheckResult {
    let start = Instant::now();
    let mut r = f();
    r.id = id.to_string();
    r.name = name.to_string();
    r.seconds = start.elapsed().as_secs_f64();
    r
}

pub fn run_selected(ctx: &Ctx) -> Vec<CheckResult> {
    let mut out = Vec::new();
    if ctx.selected("A1") {
        out.push(timed("A1", "build, clippy -D warnings, test", || a1::check(ctx)));
    } else {
        // The other checks need the binary; build quietly when A1 was deselected.
        let _ = ctx.cargo(&["build", "--workspace", "--quiet"], Duration::from_secs(600));
    }
    if ctx.selected("A2") {
        out.push(timed("A2", "keep rule through `tuner decide`", || a2::check(ctx)));
    }
    if ctx.selected("A3") {
        out.push(timed("A3", "`tuner plan`: grid, order, warm start", || a3::check(ctx)));
    }
    if ctx.selected("A4a") {
        out.push(timed("A4a", "preflight refusal on strix (reserve)", || a4::strix(ctx)));
    }
    if ctx.selected("A4b") {
        out.push(timed("A4b", "Spark sweep refused by default", || a4::spark(ctx)));
    }
    let mut recipes_from_a5: Option<PathBuf> = None;
    if ctx.selected("A5") {
        let start = Instant::now();
        let (mut r, recipes) = a5::check(ctx);
        r.id = "A5".into();
        r.name = "live sweep on strix".into();
        r.seconds = start.elapsed().as_secs_f64();
        recipes_from_a5 = recipes;
        out.push(r);
    }
    if ctx.selected("A6") {
        out.push(timed("A6", "second plan warm-starts from A5's recipe", || a6::check(ctx, recipes_from_a5.as_deref())));
    }
    if ctx.selected("A7") {
        out.push(timed("A7", "live depth sweep on strix (pp@32768)", || a7::check(ctx)));
    }
    if ctx.selected("A8") {
        out.push(timed("A8", "live KLD at 32k context on strix", || a8::check(ctx)));
    }
    out
}

pub fn fail(reason: impl Into<String>, evidence: Vec<Evidence>) -> CheckResult {
    CheckResult::new("", "", Status::Fail(reason.into())).with_evidence(evidence)
}

pub fn pass(reason: impl Into<String>, evidence: Vec<Evidence>) -> CheckResult {
    CheckResult::new("", "", Status::Pass(reason.into())).with_evidence(evidence)
}

pub fn skip(reason: impl Into<String>, evidence: Vec<Evidence>) -> CheckResult {
    CheckResult::new("", "", Status::Skip(reason.into())).with_evidence(evidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(only: &[&str]) -> Ctx {
        Ctx {
            only: only.iter().map(|s| s.to_string()).collect(),
            live: false,
            kld_32k: false,
            root: PathBuf::new(),
            tuner_bin: PathBuf::new(),
            fixtures: PathBuf::new(),
            out: PathBuf::new(),
            run_id: String::new(),
        }
    }

    #[test]
    fn only_filter_selects_families_and_members() {
        assert!(ctx(&[]).selected("A5"));
        let a4 = ctx(&["A4"]);
        assert!(a4.selected("A4a") && a4.selected("A4b") && !a4.selected("A1"));
        let a4b = ctx(&["a4b"]);
        assert!(a4b.selected("A4b") && !a4b.selected("A4a"));
    }
}
