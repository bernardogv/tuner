//! ssh to a host, with every launch in its own session so `reap` can kill it and its children
//! (llama-bench, llama-perplexity) even if this side dies.

use std::process::Command;
use std::time::Duration;

use crate::proc::{self, sh_quote, Captured};

const SSH_OPTS: [&str; 4] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"];

#[derive(Debug, Clone)]
pub struct Remote {
    pub host: String,
    pub home: String,
    pub run_dir: String,
}

fn ssh_raw(host: &str, remote_cmd: &str, stdin: Option<&str>, timeout: Duration) -> Result<Captured, String> {
    let mut cmd = Command::new("ssh");
    cmd.args(SSH_OPTS).arg(host).arg(remote_cmd);
    proc::run(cmd, stdin, timeout)
}

/// Run a read-only script on a host without creating a run dir (Spark status checks).
pub fn ssh_once(host: &str, script: &str) -> Result<Captured, String> {
    ssh_raw(host, "bash -s", Some(script), Duration::from_secs(60))
}

impl Remote {
    pub fn connect(host: &str, run_id: &str) -> Result<Remote, String> {
        let probe = ssh_raw(host, "echo \"$HOME\"", None, Duration::from_secs(60))?;
        let home = probe.stdout.trim().to_string();
        if !probe.ok() || home.is_empty() {
            return Err(format!("ssh {host} failed: {}", probe.combined().trim()));
        }
        let run_dir = format!("{home}/.tuner/runs/{run_id}");
        Ok(Remote {
            host: host.to_string(),
            home,
            run_dir,
        })
    }

    pub fn ssh(&self, script: &str, timeout: Duration) -> Result<Captured, String> {
        ssh_raw(&self.host, "bash -s", Some(script), timeout)
    }

    /// Run `script` under `setsid`, its process-group id recorded as `<run_dir>/<tag>.pgid`.
    pub fn tracked(&self, tag: &str, script: &str, timeout: Duration) -> Result<Captured, String> {
        let file = sh_quote(&format!("{}/{tag}.sh", self.run_dir));
        let pgid = sh_quote(&format!("{}/{tag}.pgid", self.run_dir));
        let remote_cmd = format!(
            "mkdir -p {} && cat > {file} && setsid -w bash -c 'echo $$ > {pgid}; exec bash {file}'",
            sh_quote(&self.run_dir)
        );
        let out = ssh_raw(&self.host, &remote_cmd, Some(script), timeout)?;
        if out.timed_out {
            let _ = self.reap();
        }
        Ok(out)
    }

    /// The first `n` bytes of a remote file, unmodified.
    pub fn head(&self, path: &str, n: usize) -> Result<Vec<u8>, String> {
        let out = Command::new("ssh")
            .args(SSH_OPTS)
            .arg(&self.host)
            .arg(format!("head -c {n} {}", sh_quote(path)))
            .output()
            .map_err(|e| format!("ssh head: {e}"))?;
        if !out.status.success() {
            return Err(format!("head {path}: {}", String::from_utf8_lossy(&out.stderr).trim()));
        }
        Ok(out.stdout)
    }

    /// TERM then KILL every process group this run started; drop the logit dumps (GBs each).
    pub fn reap(&self) -> Result<String, String> {
        let dir = sh_quote(&self.run_dir);
        let script = format!(
            r#"[ -d {dir} ] || exit 0
for sig in TERM KILL; do
  for f in {dir}/*.pgid; do [ -f "$f" ] && kill -$sig -- -"$(cat "$f")" 2>/dev/null && echo "$sig $(basename "$f")"; done
  [ $sig = TERM ] && sleep 2
done
rm -f {dir}/*.kld
true"#
        );
        Ok(self.ssh(&script, Duration::from_secs(90))?.combined())
    }
}
