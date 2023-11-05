use crate::{configure::*, types::*};

/// # Min-P sampling
/// This sampler prunes tokens that don't meet a certain percentage
/// of the most probable token. For example if `p` is `0.05` then
/// after `min_keep` is satisfied, other tokens must be at least 5%
/// of the most probable token.
///
/// Credit to @kalomaze on GitHub for design. See this link for a more in-depth
/// explanation: https://github.com/ggerganov/llama.cpp/issues/3483#issuecomment-1783920998

///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `p`: Threshold value. Use `0.0` to disable. (default: `0.9`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleMinP {
    pub(crate) p: L,
    pub(crate) min_keep: usize,
}

impl Default for SampleMinP {
    fn default() -> Self {
        Self {
            p: 0.05f32,
            min_keep: 1,
        }
    }
}

impl SampleMinP {
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

impl Sampler for SampleMinP {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let Self { p, min_keep } = *self;
        if p == 0f32 || logits.is_empty() {
            return Ok(logits);
        }

        logits.ensure_softmax()?;

        if logits.len() <= min_keep {
            return Ok(logits);
        }

        let threshold = logits[0].prob * p;
        let last_idx = logits
            .iter()
            .enumerate()
            .skip(1.max(min_keep))
            .find(|(_, l)| l.prob < threshold)
            .map(|(idx, _)| idx)
            .unwrap_or_else(|| logits.len());
        if last_idx != logits.len() {
            logits.truncate(last_idx);
            logits.set_softmax(false);
        }
        Ok(logits)
    }
}

impl ConfigurableSampler<usize, L> for SampleMinP {}

impl HasSamplerMetadata<usize, L> for SampleMinP {
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "top-p",
            description: Some(concat!(
                "This sampler adds up the token probabilities until the value is ",
                "greater or equal to p and at least min_keep tokens have been encountered.",
                " The remaining tokens are eliminated."
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "p",
                    description: Some("Target value for cumulative probabilities."),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "min_keep",
                    description: Some(concat!(
                        "Minimum number of tokens to keep after sampling. ",
                        "Setting this to 0 is not recommended."
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
                    Some(SamplerOptionValueMut::Float(&mut self.p)),
                    Some(SamplerOptionValueMut::UInt(&mut self.min_keep)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.p)),
                    Some(SamplerOptionValue::UInt(self.min_keep)),
                ],
            )
        }
    }
}
