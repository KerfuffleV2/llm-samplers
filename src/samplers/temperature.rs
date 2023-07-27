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
    const NAME: &'static str = "temperature";
    const DESC: Option<&'static str> = Some("Temperature sampling");
    const OPTIONS: &'static [SamplerOptionDefinition<Self, TID, L>] = &[SamplerOptionDefinition {
        key: "temperature",
        desc: Some("Temperature value. Higher values make the output more random."),
        typ: SamplerOptionType::Float,
        get: |slf| SamplerOptionValue::Float(slf.temperature),
        get_mut: |slf| SamplerOptionValueMut::Float(&mut slf.temperature),
    }];
}
