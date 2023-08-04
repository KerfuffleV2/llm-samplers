use std::{collections::HashMap, hash::Hash, marker::PhantomData};

use crate::{configure::*, types::*};

/// # Presence and frequency penalty sampling
/// The **presence** penalty applies to a token that appears at least once in the `last_n` tokens.
///
/// The **frequency** penalty stacks based on the number of times a token has appeared in the `last_n`
/// tokens. For example, if `frequency_penalty` is `0.05` and token id `3` has appeared 3 times in the
/// `last_n` tokens, then token id `3` will have its logit reduced by `0.05 * 3`.
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
///
/// **Parameters**:
/// - `last_n`: Number of last tokens to consider. (default: `64`)
/// - `presence_penalty`: Penalty to apply to tokens that are already present. (default: `0.0`)
/// - `frequency_penalty`: Penalty to apply to tokens based on frequency. (default: `0.0`)

#[derive(Debug, Clone)]
pub struct SampleFreqPresence<TID = u32, L = f32> {
    pub(crate) frequency_penalty: L,
    pub(crate) presence_penalty: L,
    pub(crate) last_n: usize,
    marker: PhantomData<TID>,
}

impl<TID: CanTokenId, L: CanLogit> Default for SampleFreqPresence<TID, L> {
    fn default() -> Self {
        Self {
            frequency_penalty: L::zero(),
            presence_penalty: L::zero(),
            last_n: 64,
            marker: PhantomData,
        }
    }
}

impl<TID: CanTokenId, L: CanLogit> SampleFreqPresence<TID, L> {
    pub fn new(frequency_penalty: L, presence_penalty: L, last_n: usize) -> Self {
        Self {
            frequency_penalty,
            presence_penalty,
            last_n,
            marker: PhantomData,
        }
    }

    pub fn last_n(mut self, val: usize) -> Self {
        self.last_n = val;
        self
    }

    pub fn frequency(mut self, val: L) -> Self {
        self.frequency_penalty = val;
        self
    }

    pub fn presence(mut self, val: L) -> Self {
        self.presence_penalty = val;
        self
    }
}

impl<TID: CanTokenId + Hash, L: CanLogit> Sampler<TID, L> for SampleFreqPresence<TID, L> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let Self {
            frequency_penalty,
            presence_penalty,
            last_n,
            ..
        } = *self;

        if logits.is_empty()
            || last_n == 0
            || frequency_penalty == L::zero()
            || presence_penalty == L::zero()
        {
            return Ok(logits);
        }

        let mut counts = HashMap::<TID, L>::default();

        res.with_last_tokens(&mut |orig_tokens| {
            let tokens = if last_n > orig_tokens.len() {
                orig_tokens
            } else {
                &orig_tokens[orig_tokens.len() - last_n..]
            };
            counts.reserve(tokens.len());
            tokens.iter().copied().for_each(|tid| {
                let cnt = counts.entry(tid).or_insert(L::zero());
                *cnt = *cnt + L::one()
            });
        })?;

        logits.iter_mut().for_each(|l| {
            if let Some(cnt) = counts.get(&l.token_id) {
                l.logit = l.logit
                    - (*cnt * frequency_penalty
                        + if cnt > &L::zero() {
                            L::one()
                        } else {
                            L::zero()
                        } * presence_penalty);
            }
        });
        Ok(logits.set_sorted(false))
    }
}

impl<TID, L> ConfigurableSampler<usize, L> for SampleFreqPresence<TID, L>
where
    TID: ConfigurableNumValue,
    L: ConfigurableNumValue,
{
}

impl<TID, L> HasSamplerMetadata<usize, L> for SampleFreqPresence<TID, L>
where
    TID: ConfigurableNumValue,
    L: ConfigurableNumValue,
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "frequency/presence",
            description: Some(concat!(
                "Penalty to apply to tokens based on frequency. ",
                "For example, if a token has appeared 3 times within the last_n ",
                "range then it will have its probability decreased by ",
                "3 * frequency_penalty."
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "frequency_penalty",
                    description: Some(concat!(
                        "Penalty to apply to tokens based on frequency. ",
                        "For example, if a token has appeared 3 times within the last_n ",
                        "range then it will have its probability decreased by ",
                        "3 * frequency_penalty."
                    )),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "presence_penalty",
                    description: Some(concat!(
                        "Penalty to apply to tokens that are already present ",
                        "within the last_n tokens."
                    )),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: ("last_n"),
                    description: Some(concat!(
                        "Number of previous tokens to consider when ",
                        "determining sequence repetition."
                    )),
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
                    Some(SamplerOptionValueMut::Float(&mut self.frequency_penalty)),
                    Some(SamplerOptionValueMut::Float(&mut self.presence_penalty)),
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
                    Some(SamplerOptionValue::Float(self.frequency_penalty)),
                    Some(SamplerOptionValue::Float(self.presence_penalty)),
                    Some(SamplerOptionValue::UInt(self.last_n)),
                ],
            )
        }
    }
}
