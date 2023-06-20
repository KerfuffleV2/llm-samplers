use num_traits::{Float, PrimInt};

use crate::types::*;

/// Top-P sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopP<T> {
    p: T,
    min_keep: usize,
}

impl<T: Float> SampleTopP<T> {
    pub fn new(p: T, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleTopP<L> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        use std::ops::ControlFlow::*;

        let Self { p, min_keep } = *self;
        logits.softmax();

        let mut cum_sum = L::zero();
        let last_idx =
            match logits
                .iter()
                .enumerate()
                .try_fold(logits.len(), |last_idx, (idx, logit)| {
                    cum_sum = cum_sum + logit.prob;
                    if cum_sum > p && idx >= min_keep {
                        return Break(idx);
                    }
                    Continue(last_idx)
                }) {
                Continue(i) => i,
                Break(i) => i,
            };
        logits.truncate(last_idx);
        logits
    }
}
