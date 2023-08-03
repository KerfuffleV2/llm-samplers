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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTemperature<L = f32> {
    pub(crate) temperature: L,
}

impl<L: CanLogit> Default for SampleTemperature<L> {
    fn default() -> Self {
        Self {
            temperature: L::one(),
        }
    }
}

impl<L: CanLogit> SampleTemperature<L> {
    pub fn new(temperature: L) -> Self {
        Self { temperature }
    }

    pub fn temperature(mut self, val: L) -> Self {
        self.temperature = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTemperature<L> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let temp = self.temperature;
        if temp != L::zero() {
            logits.iter_mut().for_each(|l| l.logit = l.logit / temp);
        }
        Ok(logits)
    }
}

impl<TID, L> ConfigurableSampler<TID, L> for SampleTemperature<L>
where
    TID: CanTokenId + 'static,
    L: CanLogit + 'static,
{
}

impl<UI, F> HasSamplerMetadata<UI, F> for SampleTemperature<F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
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

    fn sampler_options_mut(&mut self) -> Vec<SamplerOptionValueMut<'_, UI, F>> {
        vec![SamplerOptionValueMut::Float(&mut self.temperature)]
    }

    fn sampler_options(&self) -> Vec<SamplerOptionValue<'_, UI, F>> {
        vec![SamplerOptionValue::Float(self.temperature)]
    }
}
