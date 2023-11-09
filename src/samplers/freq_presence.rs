use std::collections::HashMap;

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
pub struct SampleFreqPresence {
    pub(crate) frequency_penalty: L,
    pub(crate) presence_penalty: L,
    pub(crate) last_n: usize,
}

impl Default for SampleFreqPresence {
    fn default() -> Self {
        Self {
            frequency_penalty: 0f32,
            presence_penalty: 0f32,
            last_n: 64,
        }
    }
}

impl SampleFreqPresence {
    pub fn new(frequency_penalty: L, presence_penalty: L, last_n: usize) -> Self {
        Self {
            frequency_penalty,
            presence_penalty,
            last_n,
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

impl Sampler for SampleFreqPresence {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let Self {
            frequency_penalty,
            presence_penalty,
            last_n,
            ..
        } = *self;

        if logits.is_empty()
            || last_n == 0
            || (frequency_penalty == 0f32 && presence_penalty == 0f32)
        {
            return Ok(logits);
        }

        let mut counts = HashMap::<TID, L>::default();
        let mut changed = 0;

        res.with_last_tokens(&mut |orig_tokens| {
            let tokens = if last_n > orig_tokens.len() {
                orig_tokens
            } else {
                &orig_tokens[orig_tokens.len() - last_n..]
            };
            counts.reserve(tokens.len());
            tokens.iter().copied().for_each(|tid| {
                let cnt = counts.entry(tid).or_insert(0f32);
                *cnt += 1f32
            });
        })?;

        logits.iter_mut().for_each(|l| {
            let Some(cnt) = counts.get(&l.token_id) else {
                return;
            };
            if cnt > &0.0 {
                l.logit -= *cnt * frequency_penalty
                    + if cnt > &0f32 { 1f32 } else { 0f32 } * presence_penalty;
                changed += 1;
            }
        });
        if changed > 0 {
            logits.set_sorted(false);
            logits.set_softmax(false);
        }
        Ok(logits)
    }
}

impl ConfigurableSampler<usize, L> for SampleFreqPresence {}

impl HasSamplerMetadata<usize, L> for SampleFreqPresence {
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
