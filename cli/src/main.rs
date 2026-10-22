use roofline::{QuantFile, active_gb, count, kv_gb, memory_gb, roofline_tok_s, weights_gb};
use roofline_benchmarks::Dataset;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage: roofline --hardware <id> --model <id> --quant <id> [--nodes N] [--ctx TOKENS] [--kv-bytes 2|1] [--data DIR]
  ids come from <DIR>/hardware.toml, <DIR>/models.toml and quants.toml next to this binary's crate
  DIR defaults to ../benchmarks/data relative to the crate";

struct Args {
    hardware: String,
    model: String,
    quant: String,
    nodes: u32,
    ctx: u64,
    kv_bytes: f64,
    data: PathBuf,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let (mut hardware, mut model, mut quant) = (None, None, None);
    let (mut nodes, mut ctx, mut kv_bytes) = (1, 0, 2.0);
    let mut data = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../benchmarks/data");
    let mut it = argv.iter();
    while let Some(flag) = it.next() {
        let val = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let num = |what: &str| val.parse::<u64>().map_err(|_| format!("{what}: not a number: {val}"));
        match flag.as_str() {
            "--hardware" => hardware = Some(val.clone()),
            "--model" => model = Some(val.clone()),
            "--quant" => quant = Some(val.clone()),
            "--nodes" => nodes = u32::try_from(num("--nodes")?).map_err(|e| e.to_string())?,
            "--ctx" => ctx = num("--ctx")?,
            "--kv-bytes" => kv_bytes = val.parse().map_err(|_| format!("--kv-bytes: {val}"))?,
            "--data" => data = PathBuf::from(val),
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if nodes == 0 {
        return Err("--nodes must be >= 1".into());
    }
    let need = |o: Option<String>, f: &str| o.ok_or_else(|| format!("{f} is required"));
    Ok(Args {
        hardware: need(hardware, "--hardware")?,
        model: need(model, "--model")?,
        quant: need(quant, "--quant")?,
        nodes,
        ctx,
        kv_bytes,
        data,
    })
}

fn run(a: &Args) -> Result<String, String> {
    let ds = Dataset::load(&a.data)?;
    let hw = ds.hardware(&a.hardware).ok_or_else(|| format!("unknown hardware {:?}", a.hardware))?;
    let model = ds.model(&a.model).ok_or_else(|| format!("unknown model {:?}", a.model))?;
    let shape = model
        .shape
        .as_ref()
        .ok_or_else(|| format!("model {:?} has no [model.shape]; add one to models.toml", a.model))?;
    let qpath = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("quants.toml");
    let text = std::fs::read_to_string(&qpath).map_err(|e| format!("{}: {e}", qpath.display()))?;
    let quants: QuantFile = toml::from_str(&text).map_err(|e| format!("{}: {e}", qpath.display()))?;
    let q = quants
        .quant
        .iter()
        .find(|q| q.id == a.quant)
        .ok_or_else(|| format!("unknown quant {:?}; known: {}", a.quant, ids(&quants)))?;

    let p = count(shape);
    let gb = active_gb(&p, q);
    let weights = weights_gb(&p, q);
    let kv = kv_gb(shape, a.ctx, a.kv_bytes);
    let mem = memory_gb(hw, a.nodes);
    let free = mem - weights - kv;
    let fit = if free >= 0.0 { "fits" } else { "DOES NOT FIT" };
    let mut out = format!(
        "model     {}  {:.2} B params, {:.2} B active\n\
         quant     {}  dense {} B/param, experts {} B/param\n\
         active    {gb:.3} GB/token  (dense {:.3} + router {:.3} + head {:.3} + experts {:.3})\n\
         weights   {weights:.1} GB",
        model.name,
        p.total() as f64 / 1e9,
        p.active() as f64 / 1e9,
        q.id,
        q.bytes_per_param,
        q.expert_bytes_per_param,
        p.dense as f64 * q.bytes_per_param / 1e9,
        p.router as f64 * 4.0 / 1e9,
        p.head as f64 * q.bytes_per_param / 1e9,
        p.experts_active as f64 * q.expert_bytes_per_param / 1e9,
    );
    if a.ctx > 0 {
        out += &format!(" + {kv:.1} GB KV at {} tokens", a.ctx);
    }
    out += &format!(
        "\nmemory    {mem:.1} GB on {} x {}: {fit}, {:.1} GB {}\n\
         roofline  {:.1} tok/s  ({} x {} GB/s / {gb:.3} GB/token), speculation off",
        a.nodes,
        hw.id,
        free.abs(),
        if free >= 0.0 { "free" } else { "short" },
        roofline_tok_s(hw, a.nodes, gb),
        a.nodes,
        hw.decode_bw_gbs,
    );
    Ok(out)
}

fn ids(q: &QuantFile) -> String {
    q.quant.iter().map(|q| q.id.as_str()).collect::<Vec<_>>().join(", ")
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(s) => {
            println!("{s}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}
