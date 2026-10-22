//! A1: the workspace builds, is clippy-clean under -D warnings, and its tests pass.

use std::time::Duration;

use super::{fail, pass, Ctx};
use crate::status::Evidence;

const STEPS: [&[&str]; 3] = [
    &["build", "--workspace"],
    &["clippy", "--workspace", "--all-targets", "--", "-D", "warnings"],
    &["test", "--workspace"],
];

pub fn check(ctx: &Ctx) -> crate::status::CheckResult {
    let mut evidence = Vec::new();
    for args in STEPS {
        let label = format!("cargo {}", args.join(" "));
        match ctx.cargo(args, Duration::from_secs(900)) {
            Ok(out) => {
                evidence.push(Evidence::new(&label, &out.combined()));
                if !out.ok() {
                    return fail(format!("{label} exited {:?}", out.status), evidence);
                }
            }
            Err(e) => {
                evidence.push(Evidence::new(&label, &e));
                return fail(format!("{label}: {e}"), evidence);
            }
        }
    }
    pass("build, clippy -D warnings and tests all clean", evidence)
}
