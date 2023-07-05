use crate::types::*;

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
pub struct SampleTemperature<L> {
    temperature: L,
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
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let temp = self.temperature;
        if temp != L::zero() {
            logits.iter_mut().for_each(|l| l.logit = l.logit / temp);
        }
        Ok(logits)
    }
}
