use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use num_traits::{Float, FromPrimitive, PrimInt};
use thiserror::Error;

pub use crate::{chain::*, resource::*};
// pub use crate::parse::*;

#[derive(Debug, Error)]
/// Sampler errors
pub enum SamplerError {
    #[error("internal error: {0}")]
    /// General internal error type.
    InternalError(String),

    #[error("missing resource error: {0}")]
    /// Missing resource error type.
    MissingResource(String),

    #[error("logits error: {0}")]
    /// Container for errors that occured while processing logits.
    LogitsError(LogitsError),

    #[error("rand error: {0}")]
    /// RNG-related errors
    RandError(rand::Error),

    #[error("rand weights error: {0}")]
    /// RNG weights-related errors
    RandWeightedError(rand::distributions::WeightedError),
}

#[derive(Debug, Clone, Error)]
/// Logit errors
pub enum LogitsError {
    #[error("Invalid logit for token id {0}")]
    /// Contains the position (AKA token id) of the offending logit.
    /// Logits cannot be NaN.
    InvalidLogit(usize),
    #[error("internal logits error: {0}")]
    /// General internal error type.
    InternalError(String),
}

impl From<LogitsError> for SamplerError {
    fn from(value: LogitsError) -> Self {
        SamplerError::LogitsError(value)
    }
}

/// Types that can be a token id implement this.
pub trait CanTokenId: PrimInt + FromPrimitive + Debug + Clone + Copy + Send + Sync {}

impl<T: PrimInt + FromPrimitive + Debug + Clone + Copy + Send + Sync> CanTokenId for T {}

/// Types that can be a logit implement this.
pub trait CanLogit: Float + FromPrimitive + Debug + Clone + Send + Sync {}

impl<T: Float + FromPrimitive + Debug + Clone + Send + Sync> CanLogit for T {}

#[derive(Debug, Clone, PartialEq)]
/// An individual logit with some additional metadata for use by the samplers.
pub struct Logit<TID = u32, L = f32> {
    /// The token id.
    pub token_id: TID,
    /// The logit value.
    pub logit: L,
    /// Computed probability.
    pub prob: L,
}

#[derive(Debug, Clone)]
/// A collection of [Logit]s. You normally will need to build this from the result of
/// evaluating the LLM.
///
/// For convenience, this can [Deref] to the internal [Vec].
pub struct Logits<TID = u32, L = f32> {
    sorted: bool,
    logits: Vec<Logit<TID, L>>,
}

impl<TID, L> Deref for Logits<TID, L> {
    type Target = Vec<Logit<TID, L>>;

    fn deref(&self) -> &Self::Target {
        &self.logits
    }
}

impl<TID, L> DerefMut for Logits<TID, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logits
    }
}

impl<TID: PrimInt, L: Float> Logits<TID, L> {
    /// Make a new [Logits<TID, L>] from an iterator of `L`. We'd like to
    /// write this as [TryFrom] but unfortunately the types make this impossible.
    pub fn try_from_iter<I: IntoIterator<Item = L>>(it: I) -> Result<Self, LogitsError> {
        let mut tid = TID::zero();
        Ok(Self {
            sorted: false,
            logits: it
                .into_iter()
                .enumerate()
                .map(|(idx, logit)| {
                    if logit.is_nan() {
                        Err(LogitsError::InvalidLogit(idx))?
                    }
                    let result = Logit {
                        token_id: tid,
                        logit,
                        prob: L::zero(),
                    };
                    tid = tid + TID::one();
                    Ok(result)
                })
                .collect::<Result<Vec<_>, LogitsError>>()?,
        })
    }
}

impl<TID: PrimInt, L: Float> TryFrom<Vec<L>> for Logits<TID, L> {
    type Error = LogitsError;

    fn try_from(value: Vec<L>) -> Result<Self, Self::Error> {
        Self::try_from_iter(value)
    }
}

impl<TID: CanTokenId, L: CanLogit> Logits<TID, L> {
    /// Get the sorted flag.
    pub fn get_sorted(&self) -> bool {
        self.sorted
    }

    /// Set the sorted flag.
    pub fn set_sorted(&mut self, is_sorted: bool) -> &mut Self {
        self.sorted = is_sorted;
        self
    }

    /// Ensure the [Logits] are sorted. Generally not necessary to call this directly.
    pub fn ensure_sorted(&mut self) -> Result<&mut Self> {
        if self.get_sorted() {
            return Ok(self);
        }

        let mut sort_err = Ok(());
        self.logits.as_mut_slice().sort_by(|a, b| {
            b.logit.partial_cmp(&a.logit).unwrap_or_else(|| {
                sort_err = Err(LogitsError::InternalError(String::from(
                    "Impossible: logit comparison failed?",
                )));
                std::cmp::Ordering::Less
            })
        });
        sort_err?;
        self.set_sorted(true);
        Ok(self)
    }

    /// Applies the softmax function to the [Logits].
    pub fn softmax(&mut self) -> Result<&mut Self> {
        if self.is_empty() {
            return Ok(self);
        }
        let max_l = if self.sorted{
            self[0].logit
        } else {
            self.iter().map(|l| l.logit).fold(L::neg_infinity(), |a, b| a.max(b))
        };
        let cum_sum = self.iter_mut().fold(L::zero(), |cs, l| {
            let p = (l.logit - max_l).exp();
            l.prob = p;
            cs + p
        });
        self.iter_mut().for_each(|l| l.prob = l.prob / cum_sum);
        Ok(self)
    }

    /// Convenience method
    pub fn sample<S: Sampler<TID, L>>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        sampler: &mut S,
    ) -> Result<&mut Self> {
        sampler.sample(res, self)
    }

    /// Convenience method
    pub fn sample_token<S: Sampler<TID, L>>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        sampler: &mut S,
    ) -> Result<Option<TID>> {
        sampler.sample_token(res, self)
    }
}

/// The main sampler trait.
pub trait Sampler<TID = u32, L = f32>: Debug + Send + Sync {
    /// Runs the [Sampler]. Depending on the type of [Sampler], this may produce a token id.
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>>;

    /// Returns the last sampled token id if available.
    ///
    /// A default implemenation is provided which simply returns [None].
    fn sampled_token_id(&self) -> Option<TID> {
        None
    }

    /// Run the sampler and return the last sampled token id if available.
    ///
    /// A default implementation is provided which just calls [Sampler::sample] followed by
    /// [Sampler::sampled_token_id()].
    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &mut Logits<TID, L>,
    ) -> Result<Option<TID>> {
        let _ = self.sample(res, logits)?;
        Ok(self.sampled_token_id())
    }
}

impl<TID, L> Sampler<TID, L> for Box<dyn Sampler<TID, L>> {
    fn sampled_token_id(&self) -> Option<TID> {
        (**self).sampled_token_id()
    }

    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &mut Logits<TID, L>,
    ) -> Result<Option<TID>> {
        (**self).sample_token(res, logits)
    }

    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>> {
        (**self).sample(res, logits)
    }
}

impl<TID, L> Sampler<TID, L> for Arc<Mutex<dyn Sampler<TID, L>>> {
    fn sampled_token_id(&self) -> Option<TID> {
        self.lock().ok()?.sampled_token_id()
    }

    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &mut Logits<TID, L>,
    ) -> Result<Option<TID>> {
        self.lock()
            .map_err(|e| SamplerError::InternalError(format!("Couldn't acquire lock: {e}")))?
            .sample_token(res, logits)
    }

    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>> {
        self.lock()
            .map_err(|e| SamplerError::InternalError(format!("Couldn't acquire lock: {e}")))?
            .sample(res, logits)
    }
}
