use rand::{
    distributions::{Distribution, WeightedIndex},
    Rng, SeedableRng,
};

use crate::types::*;

/// This might seem crazy but the idea is you could use it to do something like
/// manage global RNG state.
pub trait WithRng {
    type Rng: Rng;
    type Output;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut Self::Rng) -> Self::Output) -> Self::Output;
}

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq)]
pub struct RngBox<R>(R);

impl<R: Rng> RngBox<R> {
    pub fn new(rng: R) -> Self {
        Self(rng)
    }
}

impl<R: SeedableRng + Rng> RngBox<R> {
    pub fn new_seedable(seed: Option<u64>) -> Self {
        Self(seed.map_or_else(R::from_entropy, R::seed_from_u64))
    }
}

type WithUsizeRng<R> = dyn WithRng<Rng = R, Output = usize>;

impl<R: Rng> WithRng for RngBox<R> {
    type Rng = R;
    type Output = usize;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut R) -> Self::Output) -> Self::Output {
        fun(&mut self.0)
    }
}

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

// FIXME: Better error reporting.
// FIXME: Support logit types other than f32?
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

    fn sample_token(&mut self, logits: &mut Logits<TID, f32>) -> Option<TID> {
        self.sample(logits);
        self.get_token_id()
    }
}
