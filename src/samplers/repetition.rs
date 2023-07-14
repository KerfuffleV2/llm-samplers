use std::marker::PhantomData;

use crate::{configure::*, types::*};

// FIXME: Complete documentation.
/// # Repetition penalty sampling
/// The **repetition** penalty appears to apply to a token that has appeared at least
/// once in the `last_n` tokens. How is this different from presence penalty? Beats me!
///
/// **Properties**:
/// - Modifies logits
///
/// **Parameters**:
/// - `last_n`: Number of last tokens to consider. (default: `64`)
/// - `repetition_penalty`: Penalty to apply to repeated tokens. (default: `1.1`)
#[derive(Debug, Clone)]
pub struct SampleRepetition<TID = u32, L = f32> {
    pub(crate) repetition_penalty: L,
    pub(crate) last_n: usize,
    marker: PhantomData<TID>,
}

impl<TID: CanTokenId, L: CanLogit> Default for SampleRepetition<TID, L> {
    fn default() -> Self {
        Self {
            repetition_penalty: L::from(1.1f32).expect("Impossible: Couldn't convert f32 to Float"),
            last_n: 64,
            marker: PhantomData,
        }
    }
}

impl<TID: CanTokenId, L: CanLogit> SampleRepetition<TID, L> {
    pub fn new(repetition_penalty: L, last_n: usize) -> Self {
        Self {
            repetition_penalty,
            last_n,
            marker: PhantomData,
        }
    }

    pub fn last_n(mut self, val: usize) -> Self {
        self.last_n = val;
        self
    }

    pub fn penalty(mut self, val: L) -> Self {
        self.repetition_penalty = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleRepetition<TID, L> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let Self {
            repetition_penalty,
            last_n,
            ..
        } = *self;

        if logits.is_empty() || last_n == 0 || repetition_penalty == L::zero() {
            return Ok(logits);
        }

        res.with_last_tokens(&mut |tokens| {
            let tokens = if last_n > tokens.len() {
                tokens
            } else {
                &tokens[tokens.len() - last_n..]
            };
            logits
                .iter_mut()
                .filter(|l| tokens.contains(&l.token_id))
                .for_each(|l| {
                    l.logit = if l.logit <= L::zero() {
                        l.logit * repetition_penalty
                    } else {
                        l.logit / repetition_penalty
                    };
                });
        })?;

        Ok(logits.set_sorted(false))
    }
}

impl<TID, L> ConfigurableSampler<usize, L> for SampleRepetition<TID, L>
where
    TID: CanTokenId + 'static,
    L: CanLogit + 'static,
{
    const OPTIONS: &'static [SamplerOptionDefinition<Self, usize, L>] = &[
        SamplerOptionDefinition {
            key: "penalty",
            desc: None,
            typ: SamplerOptionType::Float,
            get: |slf| SamplerOptionValue::Float(slf.repetition_penalty),
            get_mut: |slf| SamplerOptionValueMut::Float(&mut slf.repetition_penalty),
        },
        SamplerOptionDefinition {
            key: "last_n",
            desc: None,
            typ: SamplerOptionType::UInt,
            get: |slf| SamplerOptionValue::UInt(slf.last_n),
            get_mut: |slf| SamplerOptionValueMut::UInt(&mut slf.last_n),
        },
    ];
}
