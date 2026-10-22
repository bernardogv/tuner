//! Read-only ssh probes. The harness never starts anything on a host; it only observes.

use std::process::Command;
use std::time::Duration;

use crate::proc::{self, sh_quote, Captured};

const SSH_OPTS: [&str; 4] = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20"];

/// What counts as a llama.cpp process for orphan audits and residency checks.
pub const LLAMA_PGREP: &str = "pgrep -af '^\\S*(llama-server|llama-bench|llama-perplexity)( |$)' || true";

pub fn ssh(host: &str, script: &str, timeout: Duration) -> Result<Captured, String> {
    let mut cmd = Command::new("ssh");
    cmd.args(SSH_OPTS).arg(host).arg("bash -s");
    let out = proc::run(cmd, Some(script), timeout, Duration::ZERO)?;
    if !out.ok() {
        return Err(format!("ssh {host}: {}", out.combined().trim()));
    }
    Ok(out)
}

/// One `pid cmdline` per line; empty when none.
pub fn llama_processes(host: &str) -> Result<String, String> {
    Ok(ssh(host, LLAMA_PGREP, Duration::from_secs(60))?.stdout.trim().to_string())
}

/// `*.kld` logits files under `~/.tuner/runs` modified since `since` (RFC 3339 UTC), one per line.
pub fn logits_files_since(host: &str, since: &str) -> Result<String, String> {
    let script = format!("find ~/.tuner/runs -name '*.kld' -newermt {} 2>/dev/null || true", sh_quote(since));
    Ok(ssh(host, &script, Duration::from_secs(60))?.stdout.trim().to_string())
}

#[derive(Debug, Clone, Default)]
pub struct HostState {
    pub mem_available: u64,
    /// Models llama-swap (`:8089/running`) reports as loaded.
    pub running: Vec<String>,
    pub procs: String,
    /// `None` when the GGUF is not on the host.
    pub model_bytes: Option<u64>,
    /// Free bytes on the filesystem holding `$HOME` (the tuner's run dir and logits file).
    pub disk_avail: u64,
}

/// MemAvailable, llama-swap residents, llama processes, the model's size and the free disk, in one
/// round trip.
pub fn host_state(host: &str, model: &str) -> Result<HostState, String> {
    let script = format!(
        "awk '/MemAvailable/{{print \"mem_available_kib=\" $2}}' /proc/meminfo\n\
         echo \"running=$(curl -s -m 3 localhost:8089/running || echo '{{\"running\":[]}}')\"\n\
         echo \"model_bytes=$(stat -c %s {} 2>/dev/null || echo none)\"\n\
         echo \"disk_avail=$(df -P -B1 \"$HOME\" | awk 'NR==2{{print $4}}')\"\n\
         echo procs=\n{LLAMA_PGREP}\n",
        sh_quote(model)
    );
    let out = ssh(host, &script, Duration::from_secs(60))?;
    parse_state(&out.stdout)
}

pub fn parse_state(stdout: &str) -> Result<HostState, String> {
    let mut st = HostState::default();
    let mut procs = Vec::new();
    let mut in_procs = false;
    for line in stdout.lines() {
        if in_procs {
            procs.push(line);
        } else if let Some(v) = line.strip_prefix("mem_available_kib=") {
            st.mem_available = v.trim().parse::<u64>().map_err(|e| format!("MemAvailable: {e}"))? * 1024;
        } else if let Some(v) = line.strip_prefix("running=") {
            st.running = running_models(v)?;
        } else if let Some(v) = line.strip_prefix("model_bytes=") {
            st.model_bytes = v.trim().parse().ok();
        } else if let Some(v) = line.strip_prefix("disk_avail=") {
            st.disk_avail = v.trim().parse().map_err(|e| format!("disk_avail: {e}"))?;
        } else if line == "procs=" {
            in_procs = true;
        }
    }
    st.procs = procs.join("\n").trim().to_string();
    if st.mem_available == 0 {
        return Err(format!("no MemAvailable line in: {stdout}"));
    }
    Ok(st)
}

/// llama-swap's `/running` is `{"running":[{"model":"x","state":"ready"},...]}`.
fn running_models(json: &str) -> Result<Vec<String>, String> {
    let v: serde_json::Value = serde_json::from_str(json.trim()).map_err(|e| format!("/running: {e}: {json}"))?;
    let items = v.get("running").and_then(|r| r.as_array()).ok_or_else(|| format!("/running: no array: {json}"))?;
    Ok(items
        .iter()
        .map(|i| i.get("model").and_then(|m| m.as_str()).map_or_else(|| i.to_string(), str::to_string))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_idle_and_busy_states() {
        let idle = "mem_available_kib=126353408\nrunning={\"running\":[]}\nmodel_bytes=63387346208\ndisk_avail=1184644689920\nprocs=\n";
        let st = parse_state(idle).unwrap();
        assert_eq!(st.mem_available, 126_353_408 * 1024);
        assert!(st.running.is_empty() && st.procs.is_empty());
        assert_eq!(st.model_bytes, Some(63_387_346_208));
        assert_eq!(st.disk_avail, 1_184_644_689_920);

        let busy = "mem_available_kib=50000000\nrunning={\"running\":[{\"model\":\"gpt-oss-120b-gguf\",\"state\":\"ready\"}]}\nmodel_bytes=none\nprocs=\n123 /home/x/llama-server -m y\n";
        let st = parse_state(busy).unwrap();
        assert_eq!(st.running, vec!["gpt-oss-120b-gguf".to_string()]);
        assert_eq!(st.model_bytes, None);
        assert!(st.procs.starts_with("123 "));
        assert!(parse_state("garbage").is_err());
    }
}
