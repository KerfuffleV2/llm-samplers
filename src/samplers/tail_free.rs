use num_traits::{Float, PrimInt};

pub use crate::types::*;

/// Tail free sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTailFree<T> {
    z: T,
    min_keep: usize,
}

impl<T: Float> SampleTailFree<T> {
    pub fn new(z: T, min_keep: usize) -> Self {
        Self { z, min_keep }
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleTailFree<L> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        use std::ops::ControlFlow::*;

        let Self { z, min_keep } = *self;

        if z >= L::one() || logits.len() < 2 {
            return logits;
        }

        logits.softmax();

        let mut fderivs = logits
            .iter()
            .take(logits.len() - 1)
            .enumerate()
            .map(|(idx, l)| l.prob - logits[idx + 1].prob)
            .peekable();

        let want_sderivs = logits.len() - 2;
        let mut sderivs = Vec::with_capacity(want_sderivs);
        let mut ssum = L::zero();

        while let Some(prob) = fderivs.next() {
            let sprob = (prob - *fderivs.peek().expect("ohno")).abs();
            ssum = ssum + sprob;
            sderivs.push(sprob);
            if sderivs.len() == want_sderivs {
                break;
            }
        }
        sderivs.iter_mut().for_each(|prob| *prob = *prob / ssum);

        let mut cum_sum = L::zero();
        let last_idx =
            match sderivs
                .into_iter()
                .enumerate()
                .try_fold(logits.len(), |last_idx, (idx, prob)| {
                    cum_sum = cum_sum + prob;
                    if cum_sum > z && idx >= min_keep {
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
