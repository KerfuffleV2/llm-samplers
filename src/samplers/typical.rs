use num_traits::{Float, PrimInt};

pub use crate::types::*;

/// Typical sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTypical<T> {
    p: T,
    min_keep: usize,
}

impl<T: Float> SampleTypical<T> {
    pub fn new(p: T, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleTypical<L> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        use std::ops::ControlFlow::*;

        let Self { p, min_keep } = *self;
        let min_keep = if min_keep == 0 { 0 } else { min_keep - 1 };
        logits.softmax();

        let ent = logits
            .iter()
            .fold(L::zero(), |ent, l| ent + -l.prob * l.prob.ln());

        let mut shifted = logits
            .iter()
            .map(|l| (l.clone(), (-l.prob.ln() - ent).abs()))
            .collect::<Vec<_>>();
        shifted.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Arg"));

        let mut cum_sum = L::zero();
        let last_idx = match shifted.iter().enumerate().try_fold(
            shifted.len(),
            |last_idx, (idx, (logit, _score))| {
                cum_sum = cum_sum + logit.prob;
                if cum_sum > p && idx >= min_keep {
                    return Break(idx + 1);
                }
                Continue(last_idx)
            },
        ) {
            Continue(i) => i,
            Break(i) => i,
        };
        logits.clear();
        shifted
            .into_iter()
            .take(last_idx)
            .for_each(|(logit, _score)| logits.push(logit));
        logits
    }
}
