//! A5 (`--live`, strix only): a tiny real sweep with a v0-shaped spec (depth 0, tg@0), gated by
//! the harness's own preflight. Returns the recipe dir it used so A6 can warm-start from it.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tuner::contract::{Recipe, Results, Series, Verdict, SCHEMA_VERSION};

use super::live::{self, HOST};
use super::{fail, pass, skip, Ctx, strix_gguf};
use crate::rules::{anchor_within, casebook_missing, expected_winner, validate_results, ANCHOR_TOLERANCE_PCT, STRIX_GPT_OSS_ANCHOR};
use crate::status::CheckResult;

/// Wall-clock bound for the whole sweep (two configs at depth 0 plus two KLD runs: ~3 min measured).
const SWEEP_TIMEOUT: Duration = Duration::from_secs(30 * 60);

pub fn model_file() -> &'static str {
    Path::new(strix_gguf()).file_name().and_then(|s| s.to_str()).unwrap_or(strix_gguf())
}

/// results.json parsed and schema-valid, with at least one candidate, and every KLD run's log kept
/// (contract.rs "Quality"): `kld-0.log`, `kld-control.log`, `kld-<i>.log` per benched candidate.
pub fn read_results(out_dir: &Path) -> Result<Results, String> {
    let path = out_dir.join("results.json");
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let results: Results = serde_json::from_str(&text).map_err(|e| format!("results.json does not match the schema: {e}"))?;
    let errs = validate_results(&results);
    if !errs.is_empty() {
        return Err(format!("results.json: {}", errs.join("; ")));
    }
    if results.points.is_empty() {
        return Err("no candidate points".into());
    }
    let missing = kld_logs_missing(&results, out_dir);
    if !missing.is_empty() {
        return Err(format!("KLD logs not kept in {}: {missing:?}", out_dir.display()));
    }
    Ok(results)
}

/// Names of the KLD logs the contract requires that are absent or empty.
pub fn kld_logs_missing(results: &Results, out_dir: &Path) -> Vec<String> {
    let benched = results.points.iter().enumerate().filter(|(_, p)| !p.samples.is_empty()).map(|(i, _)| format!("kld-{}", i + 1));
    ["kld-0".to_string(), "kld-control".to_string()]
        .into_iter()
        .chain(benched)
        .map(|name| format!("{name}.log"))
        .filter(|name| fs::metadata(out_dir.join(name)).map_or(true, |m| m.len() == 0))
        .collect()
}

/// The casebook draft names every series, the control and the KLD context; the recipe is fresh and
/// is the winner by `objective`.
pub fn judge_recipe_and_casebook(results: &Results, out_dir: &Path, recipes: &Path, started: &str, objective: Series, kld_ctx: u32) -> Result<Recipe, String> {
    let entry_path = out_dir.join("casebook-entry.md");
    let entry = fs::read_to_string(&entry_path).map_err(|e| format!("{}: {e}", entry_path.display()))?;
    let series: Vec<Series> = results.baseline.series.iter().map(|x| x.series).collect();
    let missing = casebook_missing(&entry, model_file().trim_end_matches(".gguf"), &series, kld_ctx);
    if !missing.is_empty() {
        return Err(format!("casebook-entry.md lacks: {missing:?}"));
    }
    let recipe_path = recipes.join(HOST).join(format!("{}.json", model_file().trim_end_matches(".gguf")));
    let recipe: Recipe = fs::read_to_string(&recipe_path)
        .map_err(|e| format!("recipe not written: {}: {e}", recipe_path.display()))
        .and_then(|t| serde_json::from_str(&t).map_err(|e| format!("recipe does not parse: {e}")))?;
    if recipe.schema_version != SCHEMA_VERSION || recipe.model != model_file() || recipe.recorded_utc.as_str() < started {
        return Err(format!("recipe is stale or wrong: {recipe:?} (run started {started})"));
    }
    let winner = expected_winner(results, objective);
    if recipe.objective != objective || recipe.winner != winner.config || recipe.series != winner.series {
        return Err(format!(
            "recipe must record the winner by {objective} ({:?}, series {:?}) with that objective; got {recipe:?}",
            winner.config, winner.series
        ));
    }
    Ok(recipe)
}

fn judge_outputs(out_dir: &Path, recipes: &Path, started: &str) -> Result<String, String> {
    let results = read_results(out_dir)?;
    // validate_results already requires baseline series; this `?` is the same guard in Result form.
    let tg = results.baseline.series.iter().find(|x| x.series == Series::TG0).map(|x| x.stats).ok_or("baseline has no tg@0")?;
    anchor_within(tg.mean, STRIX_GPT_OSS_ANCHOR, ANCHOR_TOLERANCE_PCT)?;
    let recipe = judge_recipe_and_casebook(&results, out_dir, recipes, started, Series::TG0, 512)?;
    let kept = results.points.iter().filter(|p| p.decision.verdict == Verdict::Keep).count();
    Ok(format!(
        "baseline tg@0 {:.2} tok/s ({:.1}% of roofline {:.1}), control ub {} KLD {:.4} / {:.1}%, {} candidate(s), {kept} kept, recipe winner {:?}",
        tg.mean.0,
        results.baseline.roofline_pct.map_or(0.0, |p| p.0),
        results.roofline.expected_tg_tok_s.0,
        results.control.config.ub,
        results.control.kld.mean_kld.0,
        results.control.kld.same_top_pct.0,
        results.points.len(),
        recipe.winner
    ))
}

pub fn check(ctx: &Ctx) -> (CheckResult, Option<PathBuf>) {
    if !ctx.live {
        return (skip("pass --live to run the real sweep on strix", vec![]), None);
    }
    let run = match live::sweep(ctx, "a5", "live.toml", SWEEP_TIMEOUT, live::LOGITS_BYTES_512) {
        Ok(r) => r,
        Err(result) => return (result, None),
    };
    match judge_outputs(&run.out_dir, &run.recipes, &run.started) {
        Ok(summary) => (pass(summary, run.evidence), Some(run.recipes)),
        Err(e) => (fail(e, run.evidence), None),
    }
}
