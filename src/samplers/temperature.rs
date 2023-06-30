use crate::types::*;

/// Temperature sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTemperature<T> {
    temperature: T,
}

impl<T: CanLogit> SampleTemperature<T> {
    pub fn new(temperature: T) -> Self {
        Self { temperature }
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTemperature<L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let temp = self.temperature;
        logits.iter_mut().for_each(|l| l.logit = l.logit / temp);
        Ok(logits)
    }
}
