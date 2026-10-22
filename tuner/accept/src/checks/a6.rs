//! A6: a second `plan` with A5's recipe store warm-starts from A5's winner.

use std::fs;
use std::path::Path;
use std::time::Duration;

use tuner::contract::{exit, Plan, Recipe, Spec};

use super::{describe, fail, pass, skip, Ctx, strix_gguf};
use crate::rules::check_plan;
use crate::status::{CheckResult, Evidence};

pub fn check(ctx: &Ctx, recipes: Option<&Path>) -> CheckResult {
    let Some(recipes) = recipes else {
        return skip("needs a recipe from a passing A5 (--live)", vec![]);
    };
    let recipe_path = recipes.join("strix").join("gpt-oss-120b-MXFP4.json");
    let recipe: Recipe = match fs::read_to_string(&recipe_path).map_err(|e| e.to_string()).and_then(|t| serde_json::from_str(&t).map_err(|e| e.to_string())) {
        Ok(r) => r,
        Err(e) => return fail(format!("{}: {e}", recipe_path.display()), vec![]),
    };
    let spec_path = ctx.fixture("live.toml");
    let spec: Spec = match fs::read_to_string(&spec_path).map_err(|e| e.to_string()).and_then(|t| toml::from_str(&t).map_err(|e| e.to_string())) {
        Ok(s) => s,
        Err(e) => return fail(format!("{}: {e}", spec_path.display()), vec![]),
    };
    let (spec_s, rec_s) = (spec_path.to_string_lossy().into_owned(), recipes.to_string_lossy().into_owned());
    let args = ["plan", "--host", "strix", "--model", strix_gguf(), "--spec", &spec_s, "--recipes", &rec_s];
    let out = match ctx.tuner(&args, Duration::from_secs(60), Duration::ZERO) {
        Ok(o) => o,
        Err(e) => return fail(e, vec![]),
    };
    let evidence = vec![Evidence::new(describe(&ctx.tuner_bin, &args), &out.combined())];
    if out.status != Some(exit::OK) {
        return fail(format!("plan exited {:?}", out.status), evidence);
    }
    let plan: Plan = match serde_json::from_str(&out.stdout) {
        Ok(p) => p,
        Err(e) => return fail(format!("stdout is not a Plan: {e}"), evidence),
    };
    // Contract: a recipe whose winner is the baseline (nothing was kept) adds no warm-start point.
    let warm = recipe.winner != spec.baseline;
    let errs = check_plan(&plan, &spec, warm.then_some((recipe_path.as_path(), &recipe)));
    if errs.is_empty() && !warm {
        pass("recipe winner is the baseline (nothing kept): plan correctly has no warm start".to_string(), evidence)
    } else if errs.is_empty() {
        pass(format!("plan warm-starts from {:?}", recipe.winner), evidence)
    } else {
        fail(errs.join(" | "), evidence)
    }
}
