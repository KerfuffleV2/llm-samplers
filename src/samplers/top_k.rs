use crate::{configure::*, types::*};

/// # Top-K sampling
/// This sampler retains the top `MAX(k, min_keep)` tokens
/// with the highest probability. The remaining tokens are eliminated.
///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `k`: Number of entries to keep. (default: `40`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopK {
    pub(crate) k: usize,
    pub(crate) min_keep: usize,
}

impl Default for SampleTopK {
    fn default() -> Self {
        Self { min_keep: 1, k: 40 }
    }
}

impl SampleTopK {
    pub fn new(k: usize, min_keep: usize) -> Self {
        Self { k, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn k(mut self, val: usize) -> Self {
        self.k = val;
        self
    }
}

impl<TID: CanTokenId + 'static, L: CanLogit + 'static> Sampler<TID, L> for SampleTopK {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits2: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let logits: &mut Logits<TID, L> = res.get_resource_mut("logits")?.get_resource().unwrap();
        println!("1: {logits:?}");
        println!("2: {logits2:?}");
        // let mut x = res.get_resource_mut("logits")?;
        // let logits: &mut Logits<TID, L> = (&mut x).get_resource().unwrap();
        // let mut l2 = res.with_resource_mut("logits", &mut |rv| {
        //     //
        //     None
        // });
        let k = self.k.max(self.min_keep).min(logits.len());
        logits.ensure_sorted()?.truncate(k);
        Ok(logits2)
    }
}

impl<L: ConfigurableNumValue> ConfigurableSampler<usize, L> for SampleTopK {}

impl<L: ConfigurableNumValue> HasSamplerMetadata<usize, L> for SampleTopK {
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "top-k",
            description: Some(concat!(
                "This sampler retains the top MAX(k, min_keep) tokens ",
                "with the highest probability.",
                " The remaining tokens are eliminated."
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "k",
                    description: Some("Number of tokens to keep."),
                    option_type: SamplerOptionType::UInt,
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
                HasSamplerMetadata::<usize, L>::sampler_metadata(self).options,
                [
                    Some(SamplerOptionValueMut::UInt(&mut self.k)),
                    Some(SamplerOptionValueMut::UInt(&mut self.min_keep)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                HasSamplerMetadata::<usize, L>::sampler_metadata(self).options,
                [
                    Some(SamplerOptionValue::UInt(self.k)),
                    Some(SamplerOptionValue::UInt(self.min_keep)),
                ],
            )
        }
    }
}
