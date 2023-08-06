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

        if logits.is_empty() || last_n == 0 || repetition_penalty <= L::one() {
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

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> ConfigurableSampler<usize, L>
    for SampleRepetition<TID, L>
{
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> HasSamplerMetadata<usize, L>
    for SampleRepetition<TID, L>
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "repetition",
            description: Some(concat!(
                "Applies a penalty to tokens when they've ",
                "already appeared within the previous last_n tokens."
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "penalty",
                    description: Some(
                        "Penalty to apply to tokens that meet the repetition criteria.",
                    ),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "last_n",
                    description: Some(
                        "Number of previous tokens to consider when determining repetition.",
                    ),
                    option_type: SamplerOptionType::UInt,
                },
            ],
        }
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValueMut::Float(&mut self.repetition_penalty)),
                    Some(SamplerOptionValueMut::UInt(&mut self.last_n)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.repetition_penalty)),
                    Some(SamplerOptionValue::UInt(self.last_n)),
                ],
            )
        }
    }
}
