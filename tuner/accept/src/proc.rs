//! Run a local command with a wall-clock cap, capturing both streams. On timeout the child gets
//! SIGTERM (so a tuner can reap its remote processes), then SIGKILL after `grace`.
//! Adapted from hyperloom-strix-harness/src/proc.rs.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct Captured {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
}

impl Captured {
    pub fn ok(&self) -> bool {
        self.status == Some(0) && !self.timed_out
    }

    /// Both streams, stderr last, for evidence.
    pub fn combined(&self) -> String {
        let mut s = self.stdout.clone();
        if !self.stderr.trim().is_empty() {
            if !s.is_empty() && !s.ends_with('\n') {
                s.push('\n');
            }
            s.push_str("[stderr]\n");
            s.push_str(&self.stderr);
        }
        if self.timed_out {
            s.push_str("\n[accept] command timed out and was killed\n");
        }
        s
    }
}

pub fn run(mut cmd: Command, stdin: Option<&str>, timeout: Duration, grace: Duration) -> Result<Captured, String> {
    let start = Instant::now();
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
    let mut child = cmd.spawn().map_err(|e| format!("spawn {:?}: {e}", cmd.get_program()))?;

    if let Some(text) = stdin {
        // Dropping the pipe at the end of the thread closes the child's stdin (`bash -s` needs EOF).
        if let Some(mut pipe) = child.stdin.take() {
            let text = text.to_string();
            thread::spawn(move || {
                let _ = pipe.write_all(text.as_bytes());
            });
        }
    }
    let reader = |stream: Option<Box<dyn Read + Send>>| {
        stream.map(|mut s| {
            thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = s.read_to_end(&mut buf);
                buf
            })
        })
    };
    let out_h = reader(child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>));
    let err_h = reader(child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>));

    let mut timed_out = false;
    let mut termed_at: Option<Instant> = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st.code(),
            Ok(None) => {
                if let Some(t) = termed_at {
                    if t.elapsed() > grace {
                        let _ = child.kill();
                        let _ = child.wait();
                        break None;
                    }
                } else if start.elapsed() > timeout {
                    timed_out = true;
                    if grace.is_zero() {
                        let _ = child.kill();
                        let _ = child.wait();
                        break None;
                    }
                    let _ = Command::new("kill").arg("-TERM").arg(child.id().to_string()).output();
                    termed_at = Some(Instant::now());
                }
                thread::sleep(Duration::from_millis(150));
            }
            Err(e) => return Err(format!("wait: {e}")),
        }
    };
    let collect = |h: Option<thread::JoinHandle<Vec<u8>>>| -> String {
        h.and_then(|h| h.join().ok())
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default()
    };
    Ok(Captured {
        status,
        stdout: collect(out_h),
        stderr: collect(err_h),
        timed_out,
    })
}

/// Shell-quote one word for a POSIX shell.
pub fn sh_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "-_./=:@%+~".contains(c)) {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captures_both_streams_and_status() {
        let mut c = Command::new("sh");
        c.arg("-c").arg("echo out; echo err 1>&2; exit 3");
        let r = run(c, None, Duration::from_secs(10), Duration::ZERO).unwrap();
        assert_eq!(r.status, Some(3));
        assert_eq!(r.stdout.trim(), "out");
        assert_eq!(r.stderr.trim(), "err");
        assert!(!r.ok());
    }

    #[test]
    fn feeds_stdin_and_times_out_with_term_first() {
        let r = run(Command::new("cat"), Some("hello"), Duration::from_secs(10), Duration::ZERO).unwrap();
        assert_eq!(r.stdout, "hello");
        // Traps TERM, prints a marker, exits: proves TERM (not KILL) arrived first.
        let mut slow = Command::new("sh");
        slow.arg("-c").arg("trap 'echo reaped; exit 0' TERM; while :; do sleep 0.1; done");
        let r = run(slow, None, Duration::from_millis(400), Duration::from_secs(5)).unwrap();
        assert!(r.timed_out && !r.ok());
        assert_eq!(r.stdout.trim(), "reaped");
    }

    #[test]
    fn quoting() {
        assert_eq!(sh_quote("abc/def-1.0"), "abc/def-1.0");
        assert_eq!(sh_quote("a b"), "'a b'");
        assert_eq!(sh_quote("it's"), "'it'\\''s'");
    }
}
