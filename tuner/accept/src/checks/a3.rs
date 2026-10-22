//! A3: `tuner plan` on the fixture specs: baseline first, grid count, sorted order, determinism,
//! warm start from a recipe (in-grid and off-grid), the workload echoed (v0-shaped and two-depth),
//! and a spec whose objective is at an unmeasured depth refused.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tuner::contract::{exit, Plan, Recipe, Spec};

use super::{describe, fail, pass, Ctx, strix_gguf};
use crate::proc::Captured;
use crate::rules::check_plan;
use crate::status::{CheckResult, Evidence};

fn load_recipe(dir: &Path) -> Result<(PathBuf, Recipe), String> {
    let path = dir.join("strix").join("gpt-oss-120b-MXFP4.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let r: Recipe = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((path, r))
}

fn load_spec(path: &Path) -> Result<Spec, String> {
    fs::read_to_string(path)
        .map_err(|e| e.to_string())
        .and_then(|t| toml::from_str(&t).map_err(|e| e.to_string()))
        .map_err(|e| format!("fixture {}: {e}", path.display()))
}

fn run_plan(ctx: &Ctx, spec: &str, recipes: Option<&Path>, evidence: &mut Vec<Evidence>) -> Result<Captured, String> {
    let mut args = vec!["plan", "--host", "strix", "--model", strix_gguf(), "--spec", spec];
    let rec_s;
    if let Some(r) = recipes {
        rec_s = r.to_string_lossy().into_owned();
        args.extend(["--recipes", rec_s.as_str()]);
    }
    let out = ctx.tuner(&args, Duration::from_secs(60), Duration::ZERO)?;
    evidence.push(Evidence::new(describe(&ctx.tuner_bin, &args), &out.combined()));
    Ok(out)
}

fn plan(ctx: &Ctx, spec: &str, recipes: Option<&Path>, evidence: &mut Vec<Evidence>) -> Result<(String, Plan), String> {
    let out = run_plan(ctx, spec, recipes, evidence)?;
    if out.status != Some(exit::OK) {
        return Err(format!("plan exited {:?}: {}", out.status, out.stderr.trim()));
    }
    let p: Plan = serde_json::from_str(&out.stdout).map_err(|e| format!("stdout is not a Plan: {e}"))?;
    Ok((out.stdout, p))
}

pub fn check(ctx: &Ctx) -> CheckResult {
    let spec_path = ctx.fixture("sweep.toml");
    let spec = match load_spec(&spec_path) {
        Ok(s) => s,
        Err(e) => return fail(e, vec![]),
    };
    let spec_s = spec_path.to_string_lossy().into_owned();
    let mut evidence = Vec::new();
    let mut errs = Vec::new();

    match plan(ctx, &spec_s, None, &mut evidence) {
        Ok((text, p)) => {
            errs.extend(check_plan(&p, &spec, None).into_iter().map(|e| format!("no recipe: {e}")));
            match plan(ctx, &spec_s, None, &mut evidence) {
                Ok((again, _)) if again == text => {}
                Ok(_) => errs.push("two plan runs differ byte for byte".into()),
                Err(e) => errs.push(e),
            }
        }
        Err(e) => errs.push(format!("no recipe: {e}")),
    }
    for dir in ["recipes", "recipes-offgrid"] {
        let recipes = ctx.fixture(dir);
        match (load_recipe(&recipes), plan(ctx, &spec_s, Some(&recipes), &mut evidence)) {
            (Ok((path, r)), Ok((_, p))) => {
                errs.extend(check_plan(&p, &spec, Some((&path, &r))).into_iter().map(|e| format!("{dir}: {e}")));
            }
            (Err(e), _) | (_, Err(e)) => errs.push(format!("{dir}: {e}")),
        }
    }
    // v1: a two-depth spec plans like any other and echoes its workload.
    let depth_path = ctx.fixture("live-depth.toml");
    match (load_spec(&depth_path), plan(ctx, &depth_path.to_string_lossy(), None, &mut evidence)) {
        (Ok(s), Ok((_, p))) => errs.extend(check_plan(&p, &s, None).into_iter().map(|e| format!("live-depth: {e}"))),
        (Err(e), _) | (_, Err(e)) => errs.push(format!("live-depth: {e}")),
    }
    // v1: an objective at a depth that is not measured is a bad spec, refused before any ssh.
    let bad = ctx.fixture("bad-objective.toml").to_string_lossy().into_owned();
    match run_plan(ctx, &bad, None, &mut evidence) {
        Ok(out) if out.status == Some(exit::FAILED) && out.stderr.to_lowercase().contains("objective") => {}
        Ok(out) => errs.push(format!(
            "bad-objective: expected exit {} naming the objective, got {:?}: {}",
            exit::FAILED,
            out.status,
            out.stderr.trim()
        )),
        Err(e) => errs.push(format!("bad-objective: {e}")),
    }
    if errs.is_empty() {
        pass("plan: baseline first, sorted grid, deterministic, warm start in-grid and off-grid, workload echoed, bad objective refused", evidence)
    } else {
        fail(errs.join(" | "), evidence)
    }
}
