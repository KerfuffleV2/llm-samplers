use std::cmp::Ordering;

use crate::{configure::*, types::*};

// FIXME: Complete documentation.
/// # Locally typical sampling
///
/// An approach to sampling that attempts to maximize natural
/// and human-like output.
///
/// See: <https://arxiv.org/abs/2202.00666>
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. Setting this to `0` is not recommended. (default: `1`)
/// - `p`: Referred to as τ in the paper. It suggests using 0.2
///   as a value for story generation and `0.95` for "abstractive summarization"
///   (presumably this means more factual output). (default: `1.0`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleLocallyTypical<L = f32> {
    pub(crate) p: L,
    pub(crate) min_keep: usize,
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
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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

impl<L> ConfigurableSampler<usize, L> for SampleLocallyTypical<L>
where
    L: CanLogit + 'static,
{
    const NAME: &'static str = "locally typical";
    const DESC: Option<&'static str> = Some(concat!(
        "An approach to sampling that attempts to ",
        "maximize natural and human-like output. ",
        "See: https://arxiv.org/abs/2202.00666"
    ));
    const OPTIONS: &'static [SamplerOptionDefinition<Self, usize, L>] = &[
        SamplerOptionDefinition {
            key: "p",
            desc: Some(concat!(
                "Referred to as τ in the paper. ",
                "The paper suggests 0.2 as a value for story generation ",
                "and 0.95 for \"abstractive summarization\" (",
                "presumably this means more factual output)."
            )),
            typ: SamplerOptionType::Float,
            get: |slf| SamplerOptionValue::Float(slf.p),
            get_mut: |slf| SamplerOptionValueMut::Float(&mut slf.p),
        },
        SamplerOptionDefinition {
            key: "min_keep",
            desc: Some(concat!(
                "Minimum number of tokens to keep after sampling. ",
                "Setting this to 0 is not recommended."
            )),
            typ: SamplerOptionType::UInt,
            get: |slf| SamplerOptionValue::UInt(slf.min_keep),
            get_mut: |slf| SamplerOptionValueMut::UInt(&mut slf.min_keep),
        },
    ];
}
