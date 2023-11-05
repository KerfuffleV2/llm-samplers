use std::fmt::Debug;

use crate::types::{SamplerError, TID};

/// Trait for providing resources to samplers.
pub trait HasSamplerResources: Debug {
    /// Allows a sampler to mutably access the RNG (if present).
    fn with_rng_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("rng".to_string()))
    }

    /// Allows a sampler to immutably access the last tokens (if present).
    fn with_last_tokens(&self, _fun: &mut dyn FnMut(&[TID])) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("last_tokens".to_string()))
    }

    /// Allows a sampler to mutably access the last tokens (if present).
    fn with_last_tokens_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut Vec<TID>),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("last_tokens".to_string()))
    }
}

#[derive(Debug, Clone, Default)]
/// Empty resource structure for use with samplers that don't require
/// any resources.
pub struct NilSamplerResources;

impl NilSamplerResources {
    pub fn new() -> Self {
        Self
    }
}

impl HasSamplerResources for NilSamplerResources {}

impl HasSamplerResources for () {}

/// Simple resources that can provide an RNG and/or last tokens to samplers.
pub struct SimpleSamplerResources {
    pub(crate) rng: Option<Box<dyn rand::RngCore + Send + Sync>>,

    pub(crate) last_tokens: Option<Vec<TID>>,
}

impl Debug for SimpleSamplerResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("rng", &self.rng.is_some())
            .field("last_tokens", &self.last_tokens)
            .finish()
    }
}

impl SimpleSamplerResources {
    pub fn new(
        rng: Option<Box<dyn rand::RngCore + Send + Sync>>,
        last_tokens: Option<Vec<TID>>,
    ) -> Self {
        Self { rng, last_tokens }
    }
}

impl HasSamplerResources for SimpleSamplerResources {
    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        self.rng.as_mut().map_or_else(
            || Err(SamplerError::MissingResource("rng".to_string())),
            |rng| {
                fun(rng);
                Ok(())
            },
        )
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[TID])) -> Result<(), SamplerError> {
        self.last_tokens.as_ref().map_or_else(
            || Err(SamplerError::MissingResource("last_tokens".to_string())),
            |lt| {
                fun(lt);
                Ok(())
            },
        )
    }

    fn with_last_tokens_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut Vec<TID>),
    ) -> Result<(), SamplerError> {
        self.last_tokens.as_mut().map_or_else(
            || Err(SamplerError::MissingResource("last_tokens".to_string())),
            |lt| {
                fun(lt);
                Ok(())
            },
        )
    }
}
