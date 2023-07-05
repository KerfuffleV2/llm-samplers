use std::cmp::Ordering;

use crate::types::*;

// FIXME: Complete documentation.
/// # Locally typical sampling
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `p`: TBD. (default: `1.0`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleLocallyTypical<L> {
    p: L,
    min_keep: usize,
}

impl<L: CanLogit> Default for SampleLocallyTypical<L> {
    fn default() -> Self {
        Self {
            p: L::one(),
            min_keep: 1,
        }
    }
}

impl<L: CanLogit> SampleLocallyTypical<L> {
    pub fn new(p: L, min_keep: usize) -> Self {
        Self { p, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn p(mut self, val: L) -> Self {
        self.p = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleLocallyTypical<L> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        use std::ops::ControlFlow::*;

        let Self { p, min_keep } = *self;
        let min_keep = if min_keep == 0 { 0 } else { min_keep - 1 };
        logits.softmax()?;

        let ent = logits
            .iter()
            .fold(L::zero(), |ent, l| ent + -l.prob * l.prob.ln());

        let mut shifted = logits
            .iter()
            .map(|l| (l.clone(), (-l.prob.ln() - ent).abs()))
            .collect::<Vec<_>>();
        {
            let mut sort_err = Ok(());
            shifted.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or_else(|| {
                    sort_err = Err(SamplerError::InternalError(String::from(
                        "Impossible: logit comparison failed?",
                    )));
                    Ordering::Less
                })
            });
            sort_err?;
        }

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
        Ok(logits)
    }
}
