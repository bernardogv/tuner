//! Multi-needle retrieval at long context: 8 vault codes buried in wikitext filler, 2 asked for.
//! A case passes only when both asked codes come back right. Deterministic: the same filler file
//! gives the same 24 prompts, so two arms (e.g. f16 vs q8_0 V cache) see identical inputs.

use clap::Parser;
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

const NAMES: [&str; 24] = [
    "Kestrel", "Osprey", "Heron", "Plover", "Merlin", "Falcon", "Harrier", "Kite", "Curlew", "Dunlin", "Avocet",
    "Godwit", "Lapwing", "Petrel", "Shrike", "Siskin", "Linnet", "Bunting", "Warbler", "Wren", "Thrush",
    "Starling", "Swift", "Martin",
];
const NEEDLES: usize = 8;
/// Prompt sizes in tokens; each server slot is 32k, so the largest leaves room for the answer.
const SIZES: [usize; 3] = [8_000, 16_000, 24_000];
const PER_SIZE: usize = 8;
/// Conservative for wikitext under gpt-oss's tokenizer: overshooting tokens would overflow the slot.
const CHARS_PER_TOKEN: usize = 4;

#[derive(Parser)]
struct Args {
    /// Filler text (wikitext-2 wiki.test.raw).
    #[arg(long)]
    filler: PathBuf,
    /// llama-swap model name to query.
    #[arg(long)]
    model: String,
    #[arg(long, default_value = "strix")]
    host: String,
    #[arg(long, default_value = "http://127.0.0.1:8089")]
    url: String,
    /// One JSON line per case.
    #[arg(long)]
    out: PathBuf,
}

struct Case {
    id: usize,
    tokens: usize,
    prompt: String,
    /// (name, code) for the two asked vaults.
    asked: [(String, u32); 2],
}

/// Minimal LCG (Numerical Recipes constants): no rand dependency, same sequence on every machine.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) as usize) % n
    }
}

/// Byte offset of the first sentence end (". ") at or after `at`, or `at` if none follows.
fn sentence_end(s: &str, at: usize) -> usize {
    s[at..].find(". ").map_or(at, |i| at + i + 2)
}

fn build(filler: &str, id: usize, tokens: usize) -> Case {
    let mut rng = Lcg(id as u64 + 1);
    let chars = tokens * CHARS_PER_TOKEN;
    let start = rng.next(filler.len() - chars);
    let start = (start..).find(|&i| filler.is_char_boundary(i)).unwrap_or(0);
    let end = (start + chars..).find(|&i| filler.is_char_boundary(i)).unwrap_or(filler.len());
    let mut text = filler[start..end].to_string();

    let mut pool: Vec<&str> = NAMES.to_vec();
    let needles: Vec<(String, u32)> = (0..NEEDLES)
        .map(|_| (pool.remove(rng.next(pool.len())).to_string(), 1000 + rng.next(9000) as u32))
        .collect();
    // Evenly spaced with jitter; inserted back to front so earlier offsets stay valid.
    let slot = text.len() / NEEDLES;
    let mut at: Vec<usize> = (0..NEEDLES).map(|k| k * slot + rng.next(slot / 2)).collect();
    for (k, &(ref name, code)) in needles.iter().enumerate().rev() {
        let mut p = sentence_end(&text, at[k]);
        while !text.is_char_boundary(p) {
            p += 1;
        }
        at[k] = p;
        text.insert_str(p, &format!("The access code for vault {name} is {code}. "));
    }
    // One from the first half, one from the second, so every case reaches deep into the context.
    let a = rng.next(NEEDLES / 2);
    let b = NEEDLES / 2 + rng.next(NEEDLES / 2);
    let asked = [needles[a].clone(), needles[b].clone()];
    let prompt = format!(
        "{text}\n\nThe text above contains access codes for several vaults. What are the access codes for vault {} \
         and vault {}? Reply with exactly one line: {}=<code>, {}=<code>",
        asked[0].0, asked[1].0, asked[0].0, asked[1].0
    );
    Case { id, tokens, prompt, asked }
}

/// The first run of 4 digits after `name` in `answer`.
fn code_after(answer: &str, name: &str) -> Option<u32> {
    let rest = &answer[answer.find(name)? + name.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    (digits.len() == 4).then(|| digits.parse().ok()).flatten()
}

fn score(answer: &str, asked: &[(String, u32); 2]) -> bool {
    asked.iter().all(|(name, code)| code_after(answer, name) == Some(*code))
}

fn ask(a: &Args, prompt: &str) -> Result<Value, String> {
    let body = json!({
        "model": a.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 2048,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    });
    let remote = format!(
        "curl -s -m 900 {}/v1/chat/completions -H 'Content-Type: application/json' -d @-",
        a.url
    );
    let mut child = Command::new("ssh")
        .args(["-o", "BatchMode=yes", &a.host, &remote])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ssh: {e}"))?;
    child
        .stdin
        .take()
        .ok_or("ssh stdin")?
        .write_all(body.to_string().as_bytes())
        .map_err(|e| format!("ssh stdin: {e}"))?;
    let out = child.wait_with_output().map_err(|e| format!("ssh: {e}"))?;
    serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("response: {e}: {}", String::from_utf8_lossy(&out.stderr).trim()))
}

fn main() -> Result<(), String> {
    let a = Args::parse();
    let filler = std::fs::read_to_string(&a.filler).map_err(|e| format!("{}: {e}", a.filler.display()))?;
    let mut out = std::fs::File::create(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let (mut pass, mut n) = (0, 0);
    for (s, &tokens) in SIZES.iter().enumerate() {
        for k in 0..PER_SIZE {
            let c = build(&filler, s * PER_SIZE + k, tokens);
            let t = Instant::now();
            let (answer, prompt_tokens, err) = match ask(&a, &c.prompt) {
                Ok(v) => (
                    v["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string(),
                    v["usage"]["prompt_tokens"].as_u64(),
                    v["error"].as_object().map(|e| e.get("message").map_or(Value::Null, Clone::clone)),
                ),
                Err(e) => (String::new(), None, Some(Value::String(e))),
            };
            let ok = score(&answer, &c.asked);
            pass += usize::from(ok);
            n += 1;
            let line = json!({
                "id": c.id, "tokens_target": c.tokens, "prompt_tokens": prompt_tokens,
                "asked": c.asked.iter().map(|(n, c)| format!("{n}={c}")).collect::<Vec<_>>(),
                "answer": answer, "pass": ok, "error": err, "secs": t.elapsed().as_secs_f64(),
            });
            writeln!(out, "{line}").map_err(|e| e.to_string())?;
            println!(
                "case {:2} {:>5} tok -> {} {:?} ({:.0}s)",
                c.id,
                prompt_tokens.map_or("?".into(), |p| p.to_string()),
                if ok { "PASS" } else { "FAIL" },
                answer.lines().next().unwrap_or(""),
                t.elapsed().as_secs_f64()
            );
        }
    }
    println!("{} {pass}/{n} passed", a.model);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filler() -> String {
        "Lorem ipsum dolor sit amet. ".repeat(20_000)
    }

    #[test]
    fn every_needle_lands_in_order_and_the_asked_two_are_among_them() {
        let f = filler();
        let c = build(&f, 3, 8_000);
        let found: Vec<usize> = NAMES
            .iter()
            .filter_map(|n| c.prompt.find(&format!("vault {n} is ")))
            .collect();
        assert_eq!(found.len(), NEEDLES, "all needles inserted once");
        for (name, code) in &c.asked {
            assert!(c.prompt.contains(&format!("The access code for vault {name} is {code}. ")));
        }
        assert_ne!(c.asked[0].0, c.asked[1].0);
        let first = c.prompt.find(&format!("vault {} is", c.asked[0].0)).unwrap();
        assert!(first < c.prompt.len() / 2 + 2_000, "first asked needle is in the first half");
        assert_eq!(build(&f, 3, 8_000).prompt, c.prompt, "deterministic");
    }

    #[test]
    fn scoring_needs_both_codes_against_the_right_names() {
        let asked = [("Kestrel".to_string(), 4817), ("Wren".to_string(), 2093)];
        assert!(score("Kestrel=4817, Wren=2093", &asked));
        assert!(score("**Kestrel** = 4817, **Wren** = 2093", &asked));
        assert!(!score("Kestrel=2093, Wren=4817", &asked), "swapped");
        assert!(!score("Kestrel=4817", &asked), "one missing");
        assert!(!score("Kestrel=48170, Wren=2093", &asked), "five digits");
        assert!(!score("", &asked));
    }
}
