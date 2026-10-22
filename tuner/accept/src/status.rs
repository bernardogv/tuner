//! Check outcomes: the closed `Status` enum, evidence, and the run report.
//! Copied in shape from hyperloom-strix-harness/src/status.rs.

use serde::Serialize;

/// Every arm carries a reason so a verdict is never bare.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "status", content = "reason", rename_all = "lowercase")]
pub enum Status {
    Pass(String),
    Fail(String),
    Skip(String),
}

impl Status {
    pub fn label(&self) -> &'static str {
        match self {
            Status::Pass(_) => "PASS",
            Status::Fail(_) => "FAIL",
            Status::Skip(_) => "SKIP",
        }
    }

    pub fn reason(&self) -> &str {
        match self {
            Status::Pass(r) | Status::Fail(r) | Status::Skip(r) => r,
        }
    }
}

/// One command the check ran and what it printed (trimmed).
#[derive(Debug, Clone, Serialize)]
pub struct Evidence {
    pub command: String,
    pub output: String,
}

const MAX_EVIDENCE_LINES: usize = 60;

impl Evidence {
    pub fn new(command: impl Into<String>, output: &str) -> Evidence {
        Evidence {
            command: command.into(),
            output: trim_output(output, MAX_EVIDENCE_LINES),
        }
    }
}

/// Keep the head and tail of long output.
pub fn trim_output(output: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= max_lines {
        return output.trim_end().to_string();
    }
    let head = max_lines / 3;
    let tail = max_lines - head;
    let mut out: Vec<&str> = lines[..head].to_vec();
    let omitted = format!("... [{} lines omitted] ...", lines.len() - max_lines);
    out.push(&omitted);
    out.extend_from_slice(&lines[lines.len() - tail..]);
    out.join("\n")
}

#[derive(Debug, Clone, Serialize)]
pub struct CheckResult {
    pub id: String,
    pub name: String,
    #[serde(flatten)]
    pub status: Status,
    pub evidence: Vec<Evidence>,
    pub seconds: f64,
}

impl CheckResult {
    pub fn new(id: &str, name: &str, status: Status) -> CheckResult {
        CheckResult {
            id: id.to_string(),
            name: name.to_string(),
            status,
            evidence: Vec::new(),
            seconds: 0.0,
        }
    }

    pub fn with_evidence(mut self, evidence: Vec<Evidence>) -> CheckResult {
        self.evidence = evidence;
        self
    }
}

#[derive(Debug, Serialize)]
pub struct RunReport {
    pub run_id: String,
    pub started_utc: String,
    pub live: bool,
    pub checks: Vec<CheckResult>,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
}

/// (passed, failed, skipped); the exit code is derived from `failed` alone.
pub fn tally(checks: &[CheckResult]) -> (usize, usize, usize) {
    let mut t = (0, 0, 0);
    for c in checks {
        match c.status {
            Status::Pass(_) => t.0 += 1,
            Status::Fail(_) => t.1 += 1,
            Status::Skip(_) => t.2 += 1,
        }
    }
    t
}

pub fn exit_code(checks: &[CheckResult]) -> i32 {
    if tally(checks).1 == 0 {
        0
    } else {
        1
    }
}

pub fn summary(report: &RunReport) -> String {
    let mut s = format!("accept run {} (live: {})\n", report.run_id, report.live);
    for c in &report.checks {
        s.push_str(&format!(
            "  {:<4} {:<5} {:<32} {} ({:.1}s)\n",
            c.id,
            c.status.label(),
            c.name,
            c.status.reason(),
            c.seconds
        ));
    }
    s.push_str(&format!(
        "{} passed, {} failed, {} skipped\n",
        report.passed, report.failed, report.skipped
    ));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(id: &str, status: Status) -> CheckResult {
        CheckResult::new(id, id, status)
    }

    #[test]
    fn exit_code_is_nonzero_iff_something_fails() {
        let no_fail = vec![r("A1", Status::Pass("ok".into())), r("A5", Status::Skip("busy".into()))];
        assert_eq!(exit_code(&no_fail), 0);
        let one_fail = vec![
            r("A1", Status::Pass("ok".into())),
            r("A2", Status::Fail("bad".into())),
            r("A5", Status::Skip("busy".into())),
        ];
        assert_eq!(exit_code(&one_fail), 1);
        assert_eq!(tally(&one_fail), (1, 1, 1));
    }

    #[test]
    fn trim_keeps_head_and_tail() {
        let text: String = (0..100).map(|i| format!("line{i}\n")).collect();
        let trimmed = trim_output(&text, 9);
        assert!(trimmed.starts_with("line0\nline1\nline2\n"));
        assert!(trimmed.contains("[91 lines omitted]"));
        assert!(trimmed.ends_with("line99"));
    }
}
