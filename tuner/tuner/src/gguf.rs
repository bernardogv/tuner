//! Active bytes per decoded token, from a GGUF header's tensor table (strix-casebook method,
//! `~/hip-smoke/active.py`): non-expert tensors in full + top-k/n_expert of the `*_exps` tensors;
//! `token_embd` is a one-row lookup and is not streamed, unless it doubles as the output head.
//!
//! Tensor sizes come from the gaps between data offsets (and the file end for the last one), so no
//! ggml type table is needed; alignment padding (< `general.alignment` bytes each) is noise.

use std::collections::HashMap;

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

#[derive(Debug, PartialEq)]
pub enum GgufError {
    /// The header runs past the bytes fetched; fetch more and retry.
    Truncated,
    Invalid(String),
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], GgufError> {
        let end = self.pos.checked_add(n).ok_or(GgufError::Truncated)?;
        let s = self.buf.get(self.pos..end).ok_or(GgufError::Truncated)?;
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }
    fn len(&mut self) -> Result<usize, GgufError> {
        usize::try_from(self.u64()?).map_err(|_| GgufError::Invalid("length overflows usize".into()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.len()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    /// Read one value of GGUF type `ty`; integers come back as `Some`, an array as `Some(its length)`
    /// (the token array's length is the vocab size), everything else is skipped.
    fn value(&mut self, ty: u32) -> Result<Option<u64>, GgufError> {
        let int = |b: &[u8]| {
            let mut a = [0u8; 8];
            a[..b.len()].copy_from_slice(b);
            u64::from_le_bytes(a)
        };
        match ty {
            0 | 1 | 7 => Ok(Some(int(self.take(1)?))),
            2 | 3 => Ok(Some(int(self.take(2)?))),
            4 | 5 => Ok(Some(int(self.take(4)?))),
            10 | 11 => Ok(Some(int(self.take(8)?))),
            6 => self.take(4).map(|_| None),
            12 => self.take(8).map(|_| None),
            8 => self.string().map(|_| None),
            9 => {
                let item = self.u32()?;
                let n = self.u64()?;
                for _ in 0..n {
                    self.value(item)?;
                }
                Ok(Some(n))
            }
            other => Err(GgufError::Invalid(format!("unknown value type {other}"))),
        }
    }
}

#[derive(Debug, PartialEq)]
pub struct Active {
    pub dense_bytes: u64,
    pub expert_bytes: u64,
    /// `(n_expert, n_expert_used)`; `None` for a dense model.
    pub experts: Option<(u64, u64)>,
    /// Length of the `tokenizer.ggml.tokens` array (the logits width of a KLD run); `None` when the
    /// header has no such array.
    pub n_vocab: Option<u64>,
}

impl Active {
    pub fn gb_per_token(&self) -> f64 {
        let experts = match self.experts {
            Some((n, used)) if n > 0 => self.expert_bytes as f64 * used as f64 / n as f64,
            _ => self.expert_bytes as f64,
        };
        (self.dense_bytes as f64 + experts) / 1e9
    }
}

/// `header` is the first bytes of the file (enough to cover the tensor table); `file_bytes` its size.
pub fn active(header: &[u8], file_bytes: u64) -> Result<Active, GgufError> {
    let mut c = Cursor { buf: header, pos: 0 };
    if c.take(4)? != b"GGUF" {
        return Err(GgufError::Invalid("not a GGUF file".into()));
    }
    let version = c.u32()?;
    if version < 2 {
        return Err(GgufError::Invalid(format!("GGUF v{version} unsupported")));
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    let mut ints: HashMap<String, u64> = HashMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        if let Some(v) = c.value(ty)? {
            ints.insert(key, v);
        }
    }
    let mut tensors: Vec<(String, u64)> = Vec::new();
    for _ in 0..n_tensors {
        let name = c.string()?;
        let dims = c.u32()?;
        c.take(8 * dims as usize)?;
        c.u32()?;
        tensors.push((name, c.u64()?));
    }
    let align = ints.get("general.alignment").copied().unwrap_or(32).max(1);
    let data_start = (c.pos as u64).div_ceil(align) * align;
    let data_bytes = file_bytes
        .checked_sub(data_start)
        .ok_or_else(|| GgufError::Invalid("file smaller than its header".into()))?;

    let mut by_offset: Vec<&(String, u64)> = tensors.iter().collect();
    by_offset.sort_by_key(|t| t.1);
    let has_output = tensors.iter().any(|t| t.0 == "output.weight");
    let (mut dense, mut expert) = (0u64, 0u64);
    for (i, (name, off)) in by_offset.iter().enumerate() {
        let end = by_offset.get(i + 1).map_or(data_bytes, |t| t.1);
        let size = end
            .checked_sub(*off)
            .ok_or_else(|| GgufError::Invalid(format!("tensor {name} offset past the file end")))?;
        if name.contains("_exps") {
            expert += size;
        } else if !(name == "token_embd.weight" && has_output) {
            dense += size;
        }
    }
    let arch_int = |suffix: &str| {
        ints.iter()
            .find(|(k, _)| k.ends_with(suffix) && !k.starts_with("general."))
            .map(|(_, v)| *v)
    };
    let experts = match (arch_int(".expert_count"), arch_int(".expert_used_count")) {
        (Some(n), Some(used)) => Some((n, used)),
        _ => None,
    };
    Ok(Active {
        dense_bytes: dense,
        expert_bytes: expert,
        experts,
        n_vocab: ints.get("tokenizer.ggml.tokens").copied(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(out: &mut Vec<u8>, v: &str) {
        out.extend((v.len() as u64).to_le_bytes());
        out.extend(v.as_bytes());
    }

    /// GGUF v3 with two u32 keys, one skipped array, and `tensors` as (name, offset).
    fn file(tensors: &[(&str, u64)], experts: Option<(u32, u32)>) -> Vec<u8> {
        let mut h = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend((tensors.len() as u64).to_le_bytes());
        let kv: u64 = if experts.is_some() { 3 } else { 1 };
        h.extend(kv.to_le_bytes());
        s(&mut h, "tokenizer.ggml.tokens");
        h.extend(9u32.to_le_bytes());
        h.extend(8u32.to_le_bytes());
        h.extend(2u64.to_le_bytes());
        s(&mut h, "a");
        s(&mut h, "bc");
        if let Some((n, used)) = experts {
            for (k, v) in [("gpt-oss.expert_count", n), ("gpt-oss.expert_used_count", used)] {
                s(&mut h, k);
                h.extend(4u32.to_le_bytes());
                h.extend(v.to_le_bytes());
            }
        }
        for (name, off) in tensors {
            s(&mut h, name);
            h.extend(1u32.to_le_bytes());
            h.extend(16u64.to_le_bytes());
            h.extend(0u32.to_le_bytes());
            h.extend(off.to_le_bytes());
        }
        h
    }

    fn data_start(h: &[u8]) -> u64 {
        (h.len() as u64).div_ceil(32) * 32
    }

    #[test]
    fn moe_counts_top_k_share_of_experts_and_skips_embeddings() {
        let t = [("token_embd.weight", 0), ("blk.0.ffn_up_exps.weight", 1000), ("blk.0.attn_q.weight", 129_000), ("output.weight", 130_000)];
        let h = file(&t, Some((128, 4)));
        let a = active(&h, data_start(&h) + 131_000).unwrap();
        assert_eq!(a.expert_bytes, 128_000);
        assert_eq!(a.dense_bytes, 2_000);
        assert!((a.gb_per_token() - (2_000.0 + 4_000.0) / 1e9).abs() < 1e-15);
    }

    #[test]
    fn tied_embeddings_are_streamed_as_the_output_head() {
        let t = [("token_embd.weight", 0), ("blk.0.attn_q.weight", 500)];
        let h = file(&t, None);
        let a = active(&h, data_start(&h) + 1_000).unwrap();
        assert_eq!(a.dense_bytes, 1_000);
        assert_eq!(a.experts, None);
    }

    /// v1.1: the token array's length is the vocab size the logits-file estimate needs.
    #[test]
    fn n_vocab_is_the_token_array_length() {
        let h = file(&[("output.weight", 0)], None);
        assert_eq!(active(&h, data_start(&h) + 100).unwrap().n_vocab, Some(2));
    }

    #[test]
    fn a_short_fetch_is_truncated_not_garbage() {
        let h = file(&[("output.weight", 0)], None);
        assert_eq!(active(&h[..h.len() - 3], 1 << 20), Err(GgufError::Truncated));
    }
}
