//! Standalone HTML page for the dataset: summary tiles, filterable and sortable tables,
//! a roofline bar per row, and each row's notes folded beneath it.

use crate::{Dataset, Run, Speculation, roofline_ratio, roofline_tok_s};
use std::collections::BTreeSet;
use std::fmt::Write;

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            _ => o.push(c),
        }
    }
    o
}

fn model_name(ds: &Dataset, r: &Run) -> String {
    ds.model(&r.model).map_or_else(|| r.model.clone(), |m| m.name.clone())
}

fn sorted(ds: &Dataset, pred: impl Fn(&Run) -> bool) -> Vec<&Run> {
    let mut v: Vec<&Run> = ds.runs.iter().filter(|r| pred(r)).collect();
    v.sort_by(|a, b| (&a.model, &a.date, &a.id).cmp(&(&b.model, &b.date, &b.id)));
    v
}

/// Tier a roofline percentage falls in; drives the bar colour and the word next to it.
fn tier(pct: f64) -> (&'static str, &'static str) {
    if pct < 70.0 {
        ("under", "under")
    } else if pct <= 105.0 {
        ("near", "at ceiling")
    } else {
        ("over", "spec on")
    }
}

fn bar(pct: Option<f64>) -> String {
    match pct {
        None => "<td class=\"num muted\">—</td>".into(),
        Some(p) => {
            let (cls, word) = tier(p);
            let w = (p.min(200.0) / 2.0).max(2.0);
            format!(
                "<td class=\"num\" data-v=\"{p:.1}\"><div class=\"bar {cls}\"><i style=\"width:{w:.0}%\"></i>\
                 <b>{p:.0}%</b><span>{word}</span></div></td>"
            )
        }
    }
}

fn spec_badge(s: Speculation) -> String {
    match s {
        Speculation::None => "<span class=\"tag off\">off</span>".into(),
        other => format!("<span class=\"tag spec\">{other}</span>"),
    }
}

fn engine_cell(r: &Run) -> String {
    let mut s = format!(
        "<span class=\"tag eng-{}\">{}</span><small>{}",
        r.engine.replace('.', ""),
        esc(&r.engine),
        esc(&r.engine_version)
    );
    if r.nodes > 1 {
        let _ = write!(s, " · {}× {}", r.nodes, esc(r.parallelism.as_deref().unwrap_or("")));
    }
    s.push_str("</small>");
    s
}

fn note_row(r: &Run, cols: usize) -> String {
    r.notes.as_ref().map_or_else(String::new, |n| {
        format!(
            "<tr class=\"note\" hidden><td colspan=\"{cols}\"><b>{}</b> · {}{}</td></tr>\n",
            esc(&r.id),
            esc(n.trim()),
            r.config
                .as_ref()
                .map_or_else(String::new, |c| format!("<code>{}</code>", esc(c.trim())))
        )
    })
}

fn single_stream(ds: &Dataset, out: &mut String) {
    out.push_str(
        "<h2>Single-stream decode <small>concurrency 1</small></h2>\n<table id=\"single\">\n<thead><tr>\
         <th data-k=\"s\">model</th><th data-k=\"s\">engine</th><th data-k=\"s\">quant</th><th data-k=\"s\">spec</th>\
         <th class=\"num\" data-k=\"n\">ctx</th><th class=\"num\" data-k=\"n\">decode tok/s</th>\
         <th class=\"num\" data-k=\"n\">roofline</th><th class=\"num\" data-k=\"n\">% roofline</th>\
         <th class=\"num\" data-k=\"n\">prefill tok/s</th><th data-k=\"s\">date</th></tr></thead>\n<tbody>\n",
    );
    for r in sorted(ds, |r| r.concurrency == 1) {
        let hw = ds.hardware(&r.hardware);
        let roof = hw.and_then(|h| roofline_tok_s(h, r));
        let pct = hw.and_then(|h| roofline_ratio(h, r)).map(|x| x * 100.0);
        let prefill = match (r.prefill_tok_s, r.prefill_prompt_tokens) {
            (Some(p), Some(n)) => format!("<td class=\"num\" data-v=\"{p}\">{p:.0}<small>@ {n}</small></td>"),
            _ => "<td class=\"num muted\">—</td>".into(),
        };
        let _ = writeln!(
            out,
            "<tr class=\"run\" data-model=\"{}\" data-engine=\"{}\" data-spec=\"{}\" data-has-note=\"{}\">\
             <td><b>{}</b><small>{}</small></td><td>{}</td><td>{}</td><td>{}</td>\
             <td class=\"num\" data-v=\"{}\">{}</td>\
             <td class=\"num big\" data-v=\"{}\">{:.1}</td>\
             <td class=\"num muted\" data-v=\"{}\">{}</td>{}{}<td class=\"date\">{}</td></tr>",
            esc(&r.model),
            esc(&r.engine),
            r.speculation,
            r.notes.is_some(),
            esc(&model_name(ds, r)),
            esc(&r.checkpoint),
            engine_cell(r),
            esc(&r.quant),
            spec_badge(r.speculation),
            r.context_tokens.unwrap_or(0),
            r.context_tokens.map_or("—".into(), |c| c.to_string()),
            r.decode_tok_s,
            r.decode_tok_s,
            roof.unwrap_or(0.0),
            roof.map_or("—".into(), |x| format!("{x:.1}")),
            bar(pct),
            prefill,
            r.date
        );
        out.push_str(&note_row(r, 10));
    }
    out.push_str("</tbody></table>\n");
}

fn fan_out(ds: &Dataset, out: &mut String) {
    out.push_str(
        "<h2>Fan-out decode <small>concurrency &gt; 1, aggregate tok/s</small></h2>\n<table id=\"fanout\">\n<thead><tr>\
         <th data-k=\"s\">model</th><th data-k=\"s\">engine</th><th data-k=\"s\">quant</th><th data-k=\"s\">spec</th>\
         <th class=\"num\" data-k=\"n\">conc</th><th class=\"num\" data-k=\"n\">aggregate tok/s</th>\
         <th class=\"num\" data-k=\"n\">per stream</th><th class=\"num\" data-k=\"n\">× single-stream roofline</th>\
         <th data-k=\"s\">date</th></tr></thead>\n<tbody>\n",
    );
    for r in sorted(ds, |r| r.concurrency > 1) {
        let mult = ds.hardware(&r.hardware).and_then(|h| roofline_ratio(h, r));
        let mult_cell = mult.map_or("<td class=\"num muted\">—</td>".into(), |m| {
            let w = (m.min(5.0) * 20.0).max(2.0);
            format!(
                "<td class=\"num\" data-v=\"{m:.2}\"><div class=\"bar over\"><i style=\"width:{w:.0}%\"></i><b>{m:.2}×</b></div></td>"
            )
        });
        let _ = writeln!(
            out,
            "<tr class=\"run\" data-model=\"{}\" data-engine=\"{}\" data-spec=\"{}\" data-has-note=\"{}\">\
             <td><b>{}</b><small>{}</small></td><td>{}</td><td>{}</td><td>{}</td>\
             <td class=\"num\" data-v=\"{}\">{}</td><td class=\"num big\" data-v=\"{}\">{:.1}</td>\
             <td class=\"num muted\" data-v=\"{}\">{}</td>{}<td class=\"date\">{}</td></tr>",
            esc(&r.model),
            esc(&r.engine),
            r.speculation,
            r.notes.is_some(),
            esc(&model_name(ds, r)),
            esc(&r.checkpoint),
            engine_cell(r),
            esc(&r.quant),
            spec_badge(r.speculation),
            r.concurrency,
            r.concurrency,
            r.decode_tok_s,
            r.decode_tok_s,
            r.decode_per_stream_tok_s.unwrap_or(0.0),
            r.decode_per_stream_tok_s.map_or("—".into(), |x| format!("{x:.1}")),
            mult_cell,
            r.date
        );
        out.push_str(&note_row(r, 9));
    }
    out.push_str("</tbody></table>\n");
}

fn tiles(ds: &Dataset, out: &mut String) {
    let models: BTreeSet<&str> = ds.runs.iter().map(|r| r.model.as_str()).collect();
    let engines: BTreeSet<&str> = ds.runs.iter().map(|r| r.engine.as_str()).collect();
    let best = ds
        .runs
        .iter()
        .filter(|r| r.concurrency == 1)
        .max_by(|a, b| a.decode_tok_s.total_cmp(&b.decode_tok_s));
    let (lo, hi) = ds.runs.iter().fold(("9999", "0000"), |(lo, hi), r| {
        (lo.min(r.date.as_str()), hi.max(r.date.as_str()))
    });
    let _ = writeln!(
        out,
        "<div class=\"tiles\">\
         <div><b>{}</b><span>measured cells</span></div>\
         <div><b>{}</b><span>model architectures</span></div>\
         <div><b>{}</b><span>serving engines</span></div>\
         <div><b>{:.0}</b><span>best single-stream tok/s{}</span></div>\
         <div><b>{}</b><span>to {}</span></div></div>",
        ds.runs.len(),
        models.len(),
        engines.len(),
        best.map_or(0.0, |r| r.decode_tok_s),
        best.map_or(String::new(), |r| format!(" · {}", esc(&model_name(ds, r)))),
        lo,
        hi
    );
}

fn filters(ds: &Dataset, out: &mut String) {
    let models: BTreeSet<(String, &str)> =
        ds.runs.iter().map(|r| (model_name(ds, r), r.model.as_str())).collect();
    let engines: BTreeSet<&str> = ds.runs.iter().map(|r| r.engine.as_str()).collect();
    out.push_str("<div class=\"filters\"><label>model <select data-f=\"model\"><option value=\"\">all</option>");
    for (name, id) in models {
        let _ = write!(out, "<option value=\"{}\">{}</option>", esc(id), esc(&name));
    }
    out.push_str("</select></label><label>engine <select data-f=\"engine\"><option value=\"\">all</option>");
    for e in engines {
        let _ = write!(out, "<option value=\"{0}\">{0}</option>", esc(e));
    }
    out.push_str(
        "</select></label><label>speculation <select data-f=\"spec\"><option value=\"\">all</option>\
         <option value=\"off\">off</option><option value=\"on\">on</option></select></label>\
         <label><input type=\"checkbox\" id=\"notes\"> show notes</label>\
         <span class=\"count\"></span></div>\n",
    );
}

const CSS: &str = include_str!("html.css");
const JS: &str = include_str!("html.js");

/// The whole benchmarks.html.
pub fn page(ds: &Dataset) -> String {
    let mut out = String::with_capacity(96 * 1024);
    out.push_str(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>Benchmarks · Roofline Labs</title>\n\
         <meta name=\"description\" content=\"Measured LLM decode speed on NVIDIA DGX Spark (GB10), every row scored as a percentage of the memory-bandwidth roofline.\">\n\
         <link rel=\"icon\" href=\"data:,\">\n<style>\n",
    );
    out.push_str(CSS);
    out.push_str("</style>\n</head>\n<body>\n<main>\n<p class=\"crumb\"><a href=\"./\">← Roofline Labs</a></p>\n");
    out.push_str("<h1>Benchmark table</h1>\n<p class=\"lede\">Measured LLM decode speed, model × hardware × engine × quantization, with every row scored against the hardware's memory-bandwidth roofline. Serving runs under stated configs, run by us. Nothing third-party, nothing unreproduced.</p>\n");
    tiles(ds, &mut out);
    for h in &ds.hardware {
        let _ = writeln!(
            out,
            "<section class=\"hw\"><h2>{}</h2><dl>\
             <dt>memory</dt><dd>{:.0} GiB {}</dd>\
             <dt>spec bandwidth</dt><dd>{} GB/s</dd>\
             <dt>STREAM copy</dt><dd>{} GB/s</dd>\
             <dt>decode roofline divisor</dt><dd><b>{} GB/s</b></dd></dl>\
             <p>% roofline = measured ÷ (nodes × {} ÷ active GB per token). 100% is the anchor: a clean single-stream, no-speculation run. \
             Rows above 100% have speculation on or a better kernel path. Rows well below it are dequant-compute-bound or misconfigured. \
             — means active bytes per token is not derived yet.</p></section>",
            esc(&h.name),
            h.memory_gib,
            esc(h.memory_kind.as_deref().unwrap_or("")),
            h.spec_bw_gbs,
            h.stream_copy_gbs,
            h.decode_bw_gbs,
            h.decode_bw_gbs
        );
    }
    out.push_str("<p class=\"legend\"><span class=\"sw under\"></span>under 70%<span class=\"sw near\"></span>70 to 105%<span class=\"sw over\"></span>over 105%, speculation paying · click a header to sort · click a row for its notes</p>\n");
    filters(ds, &mut out);
    single_stream(ds, &mut out);
    fan_out(ds, &mut out);
    out.push_str("<footer>Data CC-BY-4.0 · generated from the Roofline Labs benchmark dataset; do not edit by hand.</footer>\n</main>\n<script>\n");
    out.push_str(JS);
    out.push_str("</script>\n</body>\n</html>\n");
    out
}
