//! Refuse before anything starts (contract.rs "Preflight").

use crate::proc::sh_quote;
use tuner::contract::Gib;

#[derive(Debug, PartialEq)]
pub struct Preflight {
    pub mem_available: Gib,
    /// `None` when the GGUF does not exist on the host.
    pub model_bytes: Option<u64>,
    /// Models llama-swap reports resident.
    pub swap_running: Vec<String>,
    /// `pid cmdline` of llama.cpp processes already alive.
    pub procs: Vec<String>,
    /// Free bytes on the filesystem holding `$HOME` (where `~/.tuner/runs` and the logits file go).
    pub disk_avail: u64,
}

const GIB: f64 = 1_073_741_824.0;

pub fn script(model: &str) -> String {
    format!(
        r#"awk '/MemAvailable/{{print "mem_kib=" $2}}' /proc/meminfo
echo "model_bytes=$(stat -c %s {m} 2>/dev/null || echo missing)"
echo "swap=$(curl -s -m 5 http://127.0.0.1:8089/running | tr -d '\n' || true)"
echo "disk_avail=$(df -P -B1 "$HOME" | awk 'NR==2{{print $4}}')"
pgrep -af '^\S*llama-(server|bench|perplexity)( |$)' | sed 's/^/proc=/' || true"#,
        m = sh_quote(model)
    )
}

pub fn parse(out: &str) -> Result<Preflight, String> {
    let mut mem = None;
    let mut model_bytes = None;
    let mut swap_running = Vec::new();
    let mut procs = Vec::new();
    let mut disk_avail = None;
    for line in out.lines() {
        let Some((k, v)) = line.split_once('=') else { continue };
        match k {
            "mem_kib" => mem = v.trim().parse::<f64>().ok().map(|kib| Gib(kib * 1024.0 / GIB)),
            "model_bytes" => model_bytes = v.trim().parse().ok(),
            "swap" => swap_running = swap_models(v),
            "disk_avail" => disk_avail = v.trim().parse().ok(),
            "proc" => procs.push(v.to_string()),
            _ => {}
        }
    }
    Ok(Preflight {
        mem_available: mem.ok_or("no MemAvailable in preflight output")?,
        model_bytes,
        swap_running,
        procs,
        disk_avail: disk_avail.ok_or("no disk_avail in preflight output")?,
    })
}

/// `{"running":[{"model":"x",...}]}`; anything unparseable (llama-swap down) means nothing resident.
fn swap_models(json: &str) -> Vec<String> {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return Vec::new();
    };
    v.get("running")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .map(|m| m.get("model").and_then(|s| s.as_str()).unwrap_or("?").to_string())
                .collect()
        })
        .unwrap_or_default()
}

impl Preflight {
    /// `kld_base_bytes` is the baseline logits file's estimated size (`bench::kld_base_bytes`);
    /// v1.1 refuses when `disk_avail` is below it, after the memory and residency rules.
    pub fn refusal(&self, model: &str, reserve: Gib, kld_base_bytes: u64) -> Option<String> {
        let Some(bytes) = self.model_bytes else {
            return Some(format!("model {model} not found on the host"));
        };
        if !self.swap_running.is_empty() {
            return Some(format!("llama-swap holds {:?}; never load beside a resident model", self.swap_running));
        }
        if !self.procs.is_empty() {
            return Some(format!("llama.cpp already running: {}", self.procs.join("; ")));
        }
        let headroom = self.mem_available.0 - bytes as f64 / GIB;
        if headroom < reserve.0 {
            return Some(format!(
                "headroom {headroom:.1} GiB (MemAvailable {:.1} GiB - model {:.1} GiB) < reserve {:.1} GiB",
                self.mem_available.0,
                bytes as f64 / GIB,
                reserve.0
            ));
        }
        if self.disk_avail < kld_base_bytes {
            return Some(format!(
                "disk {:.1} GB free < {:.1} GB for the baseline logits file",
                self.disk_avail as f64 / 1e9,
                kld_base_bytes as f64 / 1e9
            ));
        }
        None
    }
}

/// `spark-mode status`: `box: spark2  mode: train  default: llm  MemAvailable: 33 GiB` then
/// `unit X: inactive` lines. Refuse unless the mode is the acknowledged one and every unit is inactive.
pub fn spark_refusal(status: &str, ack: &str) -> Option<String> {
    let mode = status
        .split_whitespace()
        .skip_while(|w| *w != "mode:")
        .nth(1);
    if mode != Some(ack) {
        return Some(format!("spark-mode reports mode {mode:?}, not the acknowledged {ack:?}"));
    }
    let active: Vec<&str> = status
        .lines()
        .filter(|l| l.trim_start().starts_with("unit ") && !l.trim_end().ends_with(": inactive"))
        .collect();
    if !active.is_empty() {
        return Some(format!("spark-mode units not inactive: {}", active.join("; ")));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const TB: u64 = 1_000_000_000_000;

    fn pf(mem: f64, model_gib: f64) -> Preflight {
        Preflight {
            mem_available: Gib(mem),
            model_bytes: Some((model_gib * GIB) as u64),
            swap_running: vec![],
            procs: vec![],
            disk_avail: TB,
        }
    }

    #[test]
    fn headroom_exactly_at_reserve_passes() {
        assert_eq!(pf(80.0, 60.0).refusal("m", Gib(20.0), 0), None);
        assert!(pf(79.9, 60.0).refusal("m", Gib(20.0), 0).unwrap().contains("headroom"));
    }

    #[test]
    fn a_resident_model_refuses_even_with_headroom() {
        let mut p = pf(120.0, 1.0);
        p.swap_running = swap_models(r#"{"running":[{"model":"gpt-oss","state":"ready"}]}"#);
        assert!(p.refusal("m", Gib(20.0), 0).unwrap().contains("gpt-oss"));
        assert!(swap_models(r#"{"running":[]}"#).is_empty());
        assert!(swap_models("").is_empty());
    }

    /// v1.1: a 32k logits file for gpt-oss is 13.2 GB; the host must have room for it.
    #[test]
    fn too_little_disk_for_the_logits_file_refuses() {
        let base = 13_178_306_560;
        let mut p = pf(120.0, 60.0);
        p.disk_avail = base;
        assert_eq!(p.refusal("m", Gib(20.0), base), None, "exactly enough passes");
        p.disk_avail = base - 1;
        let why = p.refusal("m", Gib(20.0), base).unwrap();
        assert!(why.contains("disk") && why.contains("13.2"), "{why}");
        assert_eq!(p.refusal("m", Gib(20.0), 3_294_576_640), None, "the 512-ctx file still fits");
    }

    #[test]
    fn parses_the_script_output() {
        let p = parse("mem_kib=125829120\nmodel_bytes=missing\nswap=\ndisk_avail=1184644689920\nproc=12 /x/llama-server -m y\n").unwrap();
        assert_eq!(p.mem_available, Gib(120.0));
        assert_eq!(p.model_bytes, None);
        assert_eq!(p.procs, vec!["12 /x/llama-server -m y"]);
        assert_eq!(p.disk_avail, 1_184_644_689_920);
        assert!(parse("mem_kib=1\nmodel_bytes=1\n").is_err(), "no disk line");
    }

    #[test]
    fn spark_mode_must_match_and_be_idle() {
        let idle = "box: spark2  mode: train  default: llm  MemAvailable: 33 GiB\nunit llm: inactive\n";
        assert_eq!(spark_refusal(idle, "train"), None);
        assert!(spark_refusal(idle, "idle").is_some());
        assert!(spark_refusal(&idle.replace(": inactive", ": active"), "train").is_some());
    }
}
