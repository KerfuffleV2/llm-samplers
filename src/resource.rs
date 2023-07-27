use std::{fmt::Debug, marker::PhantomData};

use crate::types::{CanTokenId, SamplerError};

/// Trait for providing resources to samplers.
pub trait HasSamplerResources: Debug {
    /// The token ID type for the sampler that will use these resources.
    type TokenId: Send + Sync + Clone;

    /// Allows a sampler to mutably access the RNG (if present).
    fn with_rng_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("rng".to_string()))
    }

    /// Allows a sampler to immutably access the last tokens (if present).
    fn with_last_tokens(&self, _fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("rng".to_string()))
    }

    /// Allows a sampler to mutably access the last tokens (if present).
    fn with_last_tokens_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut Vec<Self::TokenId>),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("last_tokens".to_string()))
    }
}

#[derive(Debug, Clone)]
/// Empty resource structure for use with samplers that don't require
/// any resources.
pub struct NilSamplerResources<TID = u32>(PhantomData<TID>);

impl<TID> Default for NilSamplerResources<TID> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<TID> NilSamplerResources<TID> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<TID: Debug + Send + Sync + Clone> HasSamplerResources for NilSamplerResources<TID> {
    type TokenId = TID;
}

impl HasSamplerResources for () {
    type TokenId = u32;
}

/// Simple resources that can provide an RNG and/or last tokens to samplers.
pub struct SimpleSamplerResources<TID = u32> {
    pub(crate) rng: Option<Box<dyn rand::RngCore + Send + Sync>>,

    pub(crate) last_tokens: Option<Vec<TID>>,
}

impl<TID: Debug> Debug for SimpleSamplerResources<TID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("rng", &self.rng.is_some())
            .field("last_tokens", &self.last_tokens)
            .finish()
    }
}

impl<TID: CanTokenId> SimpleSamplerResources<TID> {
    pub fn new(
        rng: Option<Box<dyn rand::RngCore + Send + Sync>>,
        last_tokens: Option<Vec<TID>>,
    ) -> Self {
        Self { rng, last_tokens }
    }
}

impl<TID: CanTokenId> HasSamplerResources for SimpleSamplerResources<TID> {
    type TokenId = TID;

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

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
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
        fun: &mut dyn FnMut(&mut Vec<Self::TokenId>),
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
