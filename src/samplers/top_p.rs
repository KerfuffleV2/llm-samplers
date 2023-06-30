use crate::types::*;

/// Top-P sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopP<T> {
    p: T,
    min_keep: usize,
}

impl<T: CanLogit> SampleTopP<T> {
    pub fn new(p: T, min_keep: usize) -> Self {
        Self { p, min_keep }
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTopP<L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        use std::ops::ControlFlow::*;

        let Self { p, min_keep } = *self;
        logits.softmax()?;

        let mut cum_sum = L::zero();
        let last_idx =
            match logits
                .iter()
                .enumerate()
                .try_fold(logits.len(), |last_idx, (idx, logit)| {
                    cum_sum = cum_sum + logit.prob;
                    if cum_sum >= p && idx + 1 >= min_keep {
                        return Break(idx + 1);
                    }
                    Continue(last_idx)
                }) {
                Continue(i) => i,
                Break(i) => i,
            };
        logits.truncate(last_idx);
        Ok(logits)
    }
}
