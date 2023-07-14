use crate::{configure::*, types::*};

/// # Top-P sampling
/// This sampler adds up the token probabilities until the value is
/// greater or equal to `p` and at least `min_keep` tokens are encountered.
/// The remaining tokens are eliminated.
///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `p`: Target value. (default: `0.9`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopP<L = f32> {
    pub(crate) p: L,
    pub(crate) min_keep: usize,
}

impl<L: CanLogit> Default for SampleTopP<L> {
    fn default() -> Self {
        Self {
            p: L::from(0.9f32).expect("Impossible: Couldn't convert f32 to Float"),
            min_keep: 1,
        }
    }
}

impl<L: CanLogit> SampleTopP<L> {
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

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTopP<L> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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

impl<L> ConfigurableSampler<usize, L> for SampleTopP<L>
where
    L: CanLogit + 'static,
{
    const OPTIONS: &'static [SamplerOptionDefinition<Self, usize, L>] = &[
        SamplerOptionDefinition {
            key: "p",
            desc: None,
            typ: SamplerOptionType::Float,
            get: |slf| SamplerOptionValue::Float(slf.p),
            get_mut: |slf| SamplerOptionValueMut::Float(&mut slf.p),
        },
        SamplerOptionDefinition {
            key: "min_keep",
            desc: None,
            typ: SamplerOptionType::UInt,
            get: |slf| SamplerOptionValue::UInt(slf.min_keep),
            get_mut: |slf| SamplerOptionValueMut::UInt(&mut slf.min_keep),
        },
    ];
}
