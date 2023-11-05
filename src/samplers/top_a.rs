use crate::{configure::*, types::*};

/// # Min-P sampling
/// This sampler prunes tokens that don't meet a threshold based
/// on the most probable token. The formula is `a1 * pow(max_prob, a2)`.
///
/// Credit to @BlinkDL on GitHub for design. See this link for a more in-depth
/// explanation: https://github.com/BlinkDL/RWKV-LM#the-top-a-sampling-method

///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `a1`: Threshold scale. Use `0.0` to disable. (default: `0.2`)
/// - `a2`: Threshold power. Use `0.0` to disable. (default: `2.0`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleTopA {
    pub(crate) a1: L,
    pub(crate) a2: L,
    pub(crate) min_keep: usize,
}

impl Default for SampleTopA {
    fn default() -> Self {
        Self {
            a1: 0.2,
            a2: 2.0,
            min_keep: 1,
        }
    }
}

impl SampleTopA {
    pub fn new(a1: L, a2: L, min_keep: usize) -> Self {
        Self { a1, a2, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn a1(mut self, val: L) -> Self {
        self.a1 = val;
        self
    }

    pub fn a2(mut self, val: L) -> Self {
        self.a2 = val;
        self
    }
}

impl Sampler for SampleTopA {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let Self { a1, a2, min_keep } = *self;
        if logits.is_empty() || a1 == 0.0 || a2 == 0.0 {
            return Ok(logits);
        }

        logits.ensure_softmax()?;

        if logits.len() <= min_keep {
            return Ok(logits);
        }

        let threshold = logits[0].prob.powf(a2) * a1;
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

impl ConfigurableSampler<usize, L> for SampleTopA {}

impl HasSamplerMetadata<usize, L> for SampleTopA {
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "top-p",
            description: Some(concat!(
                "This sampler prunes tokens that don't meet a threshold based",
                " on the most probable token. The formula is `a1 * pow(max_prob, a2)`",
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "a1",
                    description: Some("Threshold multiplier."),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "a2",
                    description: Some("Threshold power."),
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
                    Some(SamplerOptionValueMut::Float(&mut self.a1)),
                    Some(SamplerOptionValueMut::Float(&mut self.a2)),
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
                    Some(SamplerOptionValue::Float(self.a1)),
                    Some(SamplerOptionValue::Float(self.a2)),
                    Some(SamplerOptionValue::UInt(self.min_keep)),
                ],
            )
        }
    }
}
