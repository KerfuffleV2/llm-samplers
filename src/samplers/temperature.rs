use crate::{configure::*, types::*};

/// # Temperature sampling
/// **Temperature** controls how random the output is. Only relevant when using
/// samplers that utilize RNG.
///
/// **Properties**:
///
/// - Modifies logits
///
/// **Parameters**:
/// - `temperature`: Temperature value. (default: `0.8`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleTemperature {
    pub(crate) temperature: L,
}

impl Default for SampleTemperature {
    fn default() -> Self {
        Self { temperature: 1f32 }
    }
}

impl SampleTemperature {
    pub fn new(temperature: L) -> Self {
        Self { temperature }
    }

    pub fn temperature(mut self, val: L) -> Self {
        self.temperature = val;
        self
    }
}

impl Sampler for SampleTemperature {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let temp = self.temperature;
        if temp != 0f32 {
            logits.iter_mut().for_each(|l| l.logit /= temp);
            logits.set_softmax(false);
        }
        Ok(logits)
    }
}

impl<UI: ConfigurableNumValue> ConfigurableSampler<UI, L> for SampleTemperature {}

impl<UI: ConfigurableNumValue> HasSamplerMetadata<UI, L> for SampleTemperature {
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "temperature",
            description: Some("Temperature value. Higher values make the output more random."),
            options: vec![SamplerOptionMetadata {
                key: "temperature",
                description: Some("Temperature value. Higher values make the output more random."),
                option_type: SamplerOptionType::Float,
            }],
        }
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, UI, L>> {
        unsafe {
            SamplerOptions::build_options(
                HasSamplerMetadata::<UI, L>::sampler_metadata(self).options,
                [Some(SamplerOptionValueMut::Float(&mut self.temperature))],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, UI, L>> {
        unsafe {
            SamplerOptions::build_options(
                HasSamplerMetadata::<UI, L>::sampler_metadata(self).options,
                [Some(SamplerOptionValue::Float(self.temperature))],
            )
        }
    }
}
