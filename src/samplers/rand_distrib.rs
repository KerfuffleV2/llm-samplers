use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};

use crate::{rand::*, types::*};

type WithUsizeRng<R> = dyn WithRng<Rng = R, Output = usize> + Send + Sync;

/// Random distribution sampling
pub struct RandDistribSampler<TID, R> {
    rng: Box<WithUsizeRng<R>>,
    token_id: Option<TID>,
}

impl<TID: CanTokenId, R: Rng> RandDistribSampler<TID, R> {
    pub fn new(rng: Box<WithUsizeRng<R>>) -> Self {
        Self {
            token_id: None,
            rng,
        }
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId, R: Rng + Send + Sync> Sampler<TID, f32> for RandDistribSampler<TID, R> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, f32>,
    ) -> Result<&'a mut Logits<TID, f32>, SamplerError> {
        self.token_id = None;
        if logits.is_empty() {
            return Ok(logits);
        }
        logits.softmax()?;
        let dist = WeightedIndex::new(logits.iter().map(|l| l.prob))
            .map_err(SamplerError::RandWeightedError)?;
        self.token_id = Some(logits[self.rng.with_rng(&mut |r| dist.sample(r))].token_id);
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.get_token_id()
    }
}
