use num_traits::{Float, PrimInt};

use crate::types::*;

/// Temperature sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTemperature<T> {
    temperature: T,
}

impl<T: Float> SampleTemperature<T> {
    pub fn new(temperature: T) -> Self {
        Self { temperature }
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleTemperature<L> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        let temp = self.temperature;
        logits.iter_mut().for_each(|l| l.logit = l.logit / temp);
        logits
    }
}
