use crate::types::*;

/// # Tail free sampling
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `z`: TBD. (default: `1.0`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTailFree<L> {
    z: L,
    min_keep: usize,
}

impl<L: CanLogit> Default for SampleTailFree<L> {
    fn default() -> Self {
        Self {
            z: L::one(),
            min_keep: 1,
        }
    }
}

impl<L: CanLogit> SampleTailFree<L> {
    pub fn new(z: L, min_keep: usize) -> Self {
        Self { z, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn z(mut self, val: L) -> Self {
        self.z = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTailFree<L> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        use std::ops::ControlFlow::*;

        let Self { z, min_keep } = *self;

        if z >= L::one() || logits.len() < 2 {
            return Ok(logits);
        }

        logits.softmax()?;

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
            let sprob = (prob
                - *fderivs.peek().ok_or_else(|| {
                    SamplerError::InternalError(String::from(
                        "Impossible: missing next deriv item?",
                    ))
                })?)
            .abs();
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
        Ok(logits)
    }
}
