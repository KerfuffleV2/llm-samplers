use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng,
};

pub use crate::types::*;

/// Random distribution sampling
#[derive(Debug)]
pub struct RandDistribSampler<'a, TID, R> {
    rng: &'a mut R,
    token_id: Option<TID>,
}

impl<'a, TID: CanTokenId, R: Rng> RandDistribSampler<'a, TID, R> {
    pub fn new(rng: &'a mut R) -> Self {
        Self {
            token_id: None,
            rng,
        }
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

// FIXME: Better error reporting.
impl<'b, TID: CanTokenId, R: Rng> Sampler<TID, f32> for RandDistribSampler<'b, TID, R> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, f32>) -> &'a mut Logits<TID, f32> {
        self.token_id = None;
        if logits.is_empty() {
            return logits;
        }
        logits.softmax();
        if let Ok(dist) = WeightedIndex::new(logits.iter().map(|l| l.prob)) {
            self.token_id = TID::from_usize(dist.sample(self.rng));
        }
        logits
    }
}
