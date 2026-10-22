//! Grid expansion and warm start (contract.rs "Grid expansion").

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tuner::contract::{Backend, Config, Host, Plan, PlanPoint, Recipe, Role, Spec, WarmStart, SCHEMA_VERSION};

/// `hipblaslt` only means something to the HIP build.
pub fn normalize(mut c: Config) -> Config {
    match c.backend {
        Backend::Vulkan => c.hipblaslt = false,
        Backend::Hip => {}
    }
    c
}

/// Cartesian product of the listed axes over the baseline, normalized, sorted, deduplicated.
pub fn expand(spec: &Spec) -> BTreeSet<Config> {
    fn axis<T: Copy>(listed: &Option<Vec<T>>, base: T) -> Vec<T> {
        listed.clone().unwrap_or_else(|| vec![base])
    }
    let (g, b) = (&spec.grid, spec.baseline);
    let mut out = BTreeSet::new();
    for backend in axis(&g.backend, b.backend) {
        for fa in axis(&g.fa, b.fa) {
            for ub in axis(&g.ub, b.ub) {
                for bb in axis(&g.b, b.b) {
                    for ctk in axis(&g.ctk, b.ctk) {
                        for ctv in axis(&g.ctv, b.ctv) {
                            for ctx in axis(&g.ctx, b.ctx) {
                                for np in axis(&g.np, b.np) {
                                    for hipblaslt in axis(&g.hipblaslt, b.hipblaslt) {
                                        out.insert(normalize(Config {
                                            backend,
                                            fa,
                                            ub,
                                            b: bb,
                                            ctk,
                                            ctv,
                                            ctx,
                                            np,
                                            hipblaslt,
                                        }));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

pub fn model_name(model: &Path) -> String {
    model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

pub fn recipe_path(recipes: &Path, host: Host, model: &Path) -> PathBuf {
    let stem = model.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    recipes.join(host.to_string()).join(format!("{stem}.json"))
}

pub fn read_recipe(path: &Path) -> Result<Option<Recipe>, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text).map(Some).map_err(|e| format!("{}: {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(format!("{}: {e}", path.display())),
    }
}

pub fn build(spec: &Spec, host: Host, model: &Path, recipes: Option<&Path>) -> Result<Plan, String> {
    let baseline = normalize(spec.baseline);
    let warm_start = match recipes {
        Some(dir) => {
            let path = recipe_path(dir, host, model);
            read_recipe(&path)?
                .map(|r| normalize(r.winner))
                .filter(|w| *w != baseline)
                .map(|config| WarmStart { recipe: path, config })
        }
        None => None,
    };
    let mut points = vec![PlanPoint {
        role: Role::Baseline,
        config: baseline,
    }];
    if let Some(w) = &warm_start {
        points.push(PlanPoint {
            role: Role::WarmStart,
            config: w.config,
        });
    }
    let skip = warm_start.as_ref().map(|w| w.config);
    points.extend(
        expand(spec)
            .into_iter()
            .filter(|c| *c != baseline && Some(*c) != skip)
            .map(|config| PlanPoint { role: Role::Grid, config }),
    );
    Ok(Plan {
        schema_version: SCHEMA_VERSION,
        host,
        model: model.to_path_buf(),
        repeats: spec.repeats,
        workload: spec.workload.clone(),
        warm_start,
        points,
    })
}

pub fn read_spec(path: &Path) -> Result<Spec, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec: Spec = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    if spec.schema_version != SCHEMA_VERSION {
        return Err(format!("{}: schema_version {} != {SCHEMA_VERSION}", path.display(), spec.schema_version));
    }
    if spec.repeats == 0 {
        return Err(format!("{}: repeats must be >= 1", path.display()));
    }
    let w = &spec.workload;
    let unique: BTreeSet<_> = w.depths.iter().collect();
    if w.depths.is_empty() || unique.len() != w.depths.len() {
        return Err(format!("{}: workload.depths must be non-empty and unique", path.display()));
    }
    if !w.depths.contains(&w.objective.depth) {
        return Err(format!("{}: workload.objective {} is not at a listed depth", path.display(), w.objective));
    }
    if w.guard_pct.0 < 0.0 {
        return Err(format!("{}: workload.guard_pct must be >= 0", path.display()));
    }
    if crate::bench::control_config(spec.baseline).is_none() {
        return Err(format!("{}: baseline has no reorder-only control (ub 1 with b 1)", path.display()));
    }
    Ok(spec)
}
