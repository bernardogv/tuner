//! The oracle for `tuner plan` (contract.rs "Grid expansion").

use std::collections::BTreeSet;
use std::path::Path;

use tuner::contract::{Backend, Config, Plan, PlanPoint, Recipe, Role, Spec, SCHEMA_VERSION};

/// Product of the listed axes, unlisted axes at the baseline, vulkan+hipblaslt normalized,
/// baseline removed, sorted by `Ord`.
pub fn expand(spec: &Spec) -> BTreeSet<Config> {
    fn axis<T: Copy>(listed: &Option<Vec<T>>, base: T) -> Vec<T> {
        listed.clone().unwrap_or_else(|| vec![base])
    }
    fn step<T: Copy>(set: Vec<Config>, values: &[T], put: fn(&mut Config, T)) -> Vec<Config> {
        set.iter()
            .flat_map(|c| {
                values.iter().map(move |v| {
                    let mut c = *c;
                    put(&mut c, *v);
                    c
                })
            })
            .collect()
    }
    let b = spec.baseline;
    let g = &spec.grid;
    let mut set = vec![b];
    set = step(set, &axis(&g.backend, b.backend), |c, v| c.backend = v);
    set = step(set, &axis(&g.fa, b.fa), |c, v| c.fa = v);
    set = step(set, &axis(&g.ub, b.ub), |c, v| c.ub = v);
    set = step(set, &axis(&g.b, b.b), |c, v| c.b = v);
    set = step(set, &axis(&g.ctk, b.ctk), |c, v| c.ctk = v);
    set = step(set, &axis(&g.ctv, b.ctv), |c, v| c.ctv = v);
    set = step(set, &axis(&g.ctx, b.ctx), |c, v| c.ctx = v);
    set = step(set, &axis(&g.np, b.np), |c, v| c.np = v);
    set = step(set, &axis(&g.hipblaslt, b.hipblaslt), |c, v| c.hipblaslt = v);
    let mut out: BTreeSet<Config> = set
        .into_iter()
        .map(|mut c| {
            if c.backend == Backend::Vulkan {
                c.hipblaslt = false;
            }
            c
        })
        .collect();
    out.remove(&b);
    out
}

/// Every way a `Plan` can disagree with its spec and recipe.
pub fn check_plan(plan: &Plan, spec: &Spec, recipe: Option<(&Path, &Recipe)>) -> Vec<String> {
    let mut errs = Vec::new();
    if plan.schema_version != SCHEMA_VERSION {
        errs.push(format!("schema_version {} != {SCHEMA_VERSION}", plan.schema_version));
    }
    if plan.repeats != spec.repeats {
        errs.push(format!("repeats {} != spec {}", plan.repeats, spec.repeats));
    }
    if plan.workload != spec.workload {
        errs.push(format!("workload {:?} != spec {:?}", plan.workload, spec.workload));
    }
    let baseline = PlanPoint { role: Role::Baseline, config: spec.baseline };
    match plan.points.first() {
        Some(p) if *p == baseline => {}
        other => errs.push(format!("points[0] must be the baseline {baseline:?}, got {other:?}")),
    }
    let mut expected = expand(spec);
    let mut rest = plan.points.iter().skip(1).peekable();
    match recipe {
        Some((path, r)) => {
            match &plan.warm_start {
                Some(w) if w.recipe == path && w.config == r.winner => {}
                other => errs.push(format!("warm_start must be {path:?} -> {:?}, got {other:?}", r.winner)),
            }
            if r.winner != spec.baseline {
                let want = PlanPoint { role: Role::WarmStart, config: r.winner };
                match rest.next() {
                    Some(p) if *p == want => {}
                    other => errs.push(format!("points[1] must be the warm start {want:?}, got {other:?}")),
                }
                expected.remove(&r.winner);
            }
        }
        None => {
            if plan.warm_start.is_some() {
                errs.push(format!("warm_start must be null without a recipe, got {:?}", plan.warm_start));
            }
        }
    }
    let grid: Vec<Config> = rest
        .inspect(|p| {
            if p.role != Role::Grid {
                errs.push(format!("non-grid role after the head: {p:?}"));
            }
        })
        .map(|p| p.config)
        .collect();
    let want: Vec<Config> = expected.into_iter().collect();
    if grid != want {
        errs.push(format!(
            "grid points differ from the sorted expansion ({} vs {} expected; first mismatch at {:?})",
            grid.len(),
            want.len(),
            grid.iter().zip(&want).position(|(a, b)| a != b).or(Some(grid.len().min(want.len())))
        ));
    }
    errs
}

#[cfg(test)]
mod tests {
    use super::*;
    use tuner::contract::*;

    fn cfg() -> Config {
        Config {
            backend: Backend::Vulkan,
            fa: true,
            ub: 512,
            b: 2048,
            ctk: KvType::F16,
            ctv: KvType::F16,
            ctx: 8192,
            np: 1,
            hipblaslt: false,
        }
    }

    fn spec() -> Spec {
        Spec {
            schema_version: SCHEMA_VERSION,
            repeats: 3,
            workload: Workload::default(),
            baseline: cfg(),
            grid: Grid {
                backend: Some(vec![Backend::Vulkan, Backend::Hip]),
                ub: Some(vec![512, 1024]),
                hipblaslt: Some(vec![false, true]),
                ..Grid::default()
            },
        }
    }

    fn plan_for(spec: &Spec, recipe: Option<(&Path, &Recipe)>) -> Plan {
        let mut points = vec![PlanPoint { role: Role::Baseline, config: spec.baseline }];
        let mut grid = expand(spec);
        let warm_start = recipe.map(|(p, r)| {
            if r.winner != spec.baseline {
                points.push(PlanPoint { role: Role::WarmStart, config: r.winner });
                grid.remove(&r.winner);
            }
            WarmStart { recipe: p.to_path_buf(), config: r.winner }
        });
        points.extend(grid.into_iter().map(|config| PlanPoint { role: Role::Grid, config }));
        Plan {
            schema_version: SCHEMA_VERSION,
            host: Host::Strix,
            model: "m.gguf".into(),
            repeats: spec.repeats,
            workload: spec.workload.clone(),
            warm_start,
            points,
        }
    }

    #[test]
    fn expansion_normalizes_and_drops_the_baseline() {
        // backend 2 x ub 2 x hipblaslt 2 = 8; vulkan collapses hipblaslt -> 6; minus baseline -> 5.
        let set = expand(&spec());
        assert_eq!(set.len(), 5);
        assert!(!set.contains(&cfg()));
        assert!(set.iter().all(|c| c.backend == Backend::Hip || !c.hipblaslt));
        let empty = Spec { grid: Grid::default(), ..spec() };
        assert!(expand(&empty).is_empty());
    }

    #[test]
    fn plan_checks_catch_every_disagreement() {
        let s = spec();
        let good = plan_for(&s, None);
        assert!(check_plan(&good, &s, None).is_empty(), "{:?}", check_plan(&good, &s, None));

        let mut unsorted = good.clone();
        unsorted.points.swap(1, 2);
        assert!(!check_plan(&unsorted, &s, None).is_empty(), "order matters");

        let mut baseline_last = good.clone();
        baseline_last.points.rotate_left(1);
        assert!(check_plan(&baseline_last, &s, None).iter().any(|e| e.contains("points[0]")));

        let mut short = good.clone();
        short.points.pop();
        assert!(!check_plan(&short, &s, None).is_empty(), "count matters");

        let mut dup = good.clone();
        dup.points.push(dup.points[1].clone());
        assert!(!check_plan(&dup, &s, None).is_empty(), "duplicates matter");

        let mut repeats = good.clone();
        repeats.repeats = 1;
        assert!(check_plan(&repeats, &s, None).iter().any(|e| e.contains("repeats")));

        let mut workload = good.clone();
        workload.workload.depths.push(Depth(32768));
        assert!(check_plan(&workload, &s, None).iter().any(|e| e.contains("workload")), "the plan must echo the spec's workload");
    }

    #[test]
    fn plan_warm_start_in_grid_and_off_grid() {
        let s = spec();
        let path = Path::new("recipes/strix/m.json");
        let mut r = Recipe {
            schema_version: SCHEMA_VERSION,
            host: Host::Strix,
            model: "m.gguf".into(),
            winner: Config { backend: Backend::Hip, ub: 1024, hipblaslt: true, ..cfg() },
            objective: Series::TG0,
            series: vec![],
            recorded_utc: "2026-10-18T00:00:00Z".into(),
        };
        let in_grid = plan_for(&s, Some((path, &r)));
        assert_eq!(in_grid.points.len(), 1 + 5, "warm start moved to the front, not added");
        assert_eq!(in_grid.points[1].role, Role::WarmStart);
        assert!(check_plan(&in_grid, &s, Some((path, &r))).is_empty());
        assert!(!check_plan(&in_grid, &s, None).is_empty(), "a plan with a warm start but no recipe is wrong");
        assert!(!check_plan(&plan_for(&s, None), &s, Some((path, &r))).is_empty(), "recipe ignored");

        r.winner.ctx = 16384;
        let off_grid = plan_for(&s, Some((path, &r)));
        assert_eq!(off_grid.points.len(), 1 + 1 + 5, "off-grid warm start is inserted");
        assert!(check_plan(&off_grid, &s, Some((path, &r))).is_empty());

        r.winner = cfg();
        let same = plan_for(&s, Some((path, &r)));
        assert_eq!(same.points.len(), 1 + 5, "winner == baseline adds no point");
        assert!(check_plan(&same, &s, Some((path, &r))).is_empty());
    }
}
