use rand::{
    distributions::{Distribution, WeightedIndex},
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::types::*;

/// This might seem crazy but the idea is you could use it to do something like
/// manage global RNG state.
pub trait WithRng<R: Rng> {
    type Output;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut R) -> Self::Output) -> Self::Output;
}

#[repr(transparent)]
pub struct RngBox<R>(R);

impl RngBox<StdRng> {
    pub fn new(seed: Option<u64>) -> Self {
        Self(seed.map_or_else(StdRng::from_entropy, StdRng::seed_from_u64))
    }
}

type WithUsizeRng<R, A> = dyn WithRng<R, Output = A>;

impl<R: Rng> WithRng<R> for RngBox<R> {
    type Output = usize;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut R) -> Self::Output) -> Self::Output {
        fun(&mut self.0)
    }
}

/// Random distribution sampling
pub struct RandDistribSampler<TID, R> {
    rng: Box<WithUsizeRng<R, usize>>,
    token_id: Option<TID>,
}

impl<TID: CanTokenId, R: Rng> RandDistribSampler<TID, R> {
    pub fn new(rng: Box<WithUsizeRng<R, usize>>) -> Self {
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
impl<TID: CanTokenId, R: Rng> Sampler<TID, f32> for RandDistribSampler<TID, R> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, f32>) -> &'a mut Logits<TID, f32> {
        self.token_id = None;
        if logits.is_empty() {
            return logits;
        }
        logits.softmax();
        if let Ok(dist) = WeightedIndex::new(logits.iter().map(|l| l.prob)) {
            self.token_id = Some(logits[self.rng.with_rng(&mut |r| dist.sample(r))].token_id);
        }
        logits
    }
}
