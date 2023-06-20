use num_traits::{Float, PrimInt};

pub use crate::types::*;

/// Top-K sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopK {
    k: usize,
    min_keep: usize,
}

impl SampleTopK {
    pub fn new(k: usize, min_keep: usize) -> Self {
        Self { k, min_keep }
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleTopK {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        let k = self.k.max(self.min_keep).min(logits.len());
        logits.ensure_sorted().truncate(k);
        logits
    }
}
