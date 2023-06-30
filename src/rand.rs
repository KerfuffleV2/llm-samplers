use rand::{rngs::StdRng, Rng, SeedableRng};
use std::{
    marker::PhantomData,
    sync::{Arc, Mutex},
};

/// This might seem crazy but the idea is you could use it to do something like
/// manage global RNG state.
pub trait WithRng {
    /// RNG associated type.
    type Rng: Rng;
    /// Output associated type. Unfortunately this is necessary because you
    /// can't have generics in trait objects.
    type Output;

    /// Apply a function with the RNG.
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut Self::Rng) -> Self::Output) -> Self::Output;
}

#[derive(Debug, Clone, PartialEq)]
/// A box to hold a RNG. You generally won't be able to clone or use this in multiple threads.
/// For that use case, look at [SyncRngBox].
pub struct RngBox<R = StdRng, O = usize>(R, PhantomData<*const O>);

impl<R: Rng, O> RngBox<R, O> {
    pub fn new(rng: R) -> Self {
        Self(rng, PhantomData::default())
    }
}

impl<R: SeedableRng + Rng, O> RngBox<R, O> {
    /// Create a [RngBox] from a seedable RNG. If the seed isn't specified,
    /// the method will use available entropy to seed the RNG.
    pub fn new_seedable(seed: Option<u64>) -> Self {
        Self::new(seed.map_or_else(R::from_entropy, R::seed_from_u64))
    }
}

impl<R: SeedableRng + Rng, O> Default for RngBox<R, O> {
    /// Same as calling [RngBox::new_seenable(None)].
    fn default() -> Self {
        Self::new_seedable(None)
    }
}

impl<R: Rng, O> WithRng for RngBox<R, O> {
    type Rng = R;
    type Output = O;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut R) -> Self::Output) -> Self::Output {
        fun(&mut self.0)
    }
}

unsafe impl<R: Send, O: Send> Send for RngBox<R, O> {}
unsafe impl<R: Sync, O: Sync> Sync for RngBox<R, O> {}

#[derive(Debug, Clone)]
/// Thread safe box to hold an RNG.
pub struct SyncRngBox<R = StdRng, O = usize>(Arc<Mutex<R>>, PhantomData<*const O>);

impl<R: Rng, O> SyncRngBox<R, O> {
    pub fn new(rng: R) -> Self {
        Self(Arc::new(Mutex::new(rng)), PhantomData::default())
    }
}

impl<R: SeedableRng + Rng, O> SyncRngBox<R, O> {
    pub fn new_seedable(seed: Option<u64>) -> Self {
        Self::new(seed.map_or_else(R::from_entropy, R::seed_from_u64))
    }
}

impl<R: SeedableRng + Rng, O> Default for SyncRngBox<R, O> {
    fn default() -> Self {
        Self::new_seedable(None)
    }
}

impl<R: Rng, O> WithRng for SyncRngBox<R, O> {
    type Rng = R;
    type Output = O;
    fn with_rng(&mut self, fun: &mut dyn FnMut(&mut R) -> Self::Output) -> Self::Output {
        fun(&mut self.0.lock().expect("Mutex fail"))
    }
}

unsafe impl<R: Send, O: Send> Send for SyncRngBox<R, O> {}
unsafe impl<R: Sync, O: Sync> Sync for SyncRngBox<R, O> {}
