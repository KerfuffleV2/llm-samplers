use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use thiserror::Error;

pub use crate::{chain::*, resource::*};

/// Type for token IDs.
pub type TID = u32;

/// Type for logits.
pub type L = f32;

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

#[derive(Debug, Clone, PartialEq)]
/// An individual logit with some additional metadata for use by the samplers.
pub struct Logit {
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
pub struct Logits {
    sorted: bool,
    logits: Vec<Logit>,
}

impl Deref for Logits {
    type Target = Vec<Logit>;

    fn deref(&self) -> &Self::Target {
        &self.logits
    }
}

impl DerefMut for Logits {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logits
    }
}

impl Logits {
    /// Make a new [Logits] from an iterator of `L`. We'd like to
    /// write this as [TryFrom] but unfortunately the types make this impossible.
    pub fn try_from_iter<I: IntoIterator<Item = L>>(it: I) -> Result<Self, LogitsError> {
        let mut tid = 0;
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
                        prob: 0f32,
                    };
                    tid += 1;
                    Ok(result)
                })
                .collect::<Result<Vec<_>, LogitsError>>()?,
        })
    }
}

impl TryFrom<Vec<L>> for Logits {
    type Error = LogitsError;

    fn try_from(value: Vec<L>) -> Result<Self, Self::Error> {
        Self::try_from_iter(value)
    }
}

impl Logits {
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
        self.ensure_sorted()?;
        let max_l = self[0].logit;
        let cum_sum = self.iter_mut().fold(0f32, |cs, l| {
            let p = (l.logit - max_l).exp();
            l.prob = p;
            cs + p
        });
        self.iter_mut().for_each(|l| l.prob /= cum_sum);
        Ok(self)
    }

    /// Convenience method
    pub fn sample<S: Sampler>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        sampler: &mut S,
    ) -> Result<&mut Self> {
        sampler.sample(res, self)
    }

    /// Convenience method
    pub fn sample_token<S: Sampler>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        sampler: &mut S,
    ) -> Result<Option<TID>> {
        sampler.sample_token(res, self)
    }
}

/// The main sampler trait.
pub trait Sampler: Debug + Send + Sync {
    /// Runs the [Sampler]. Depending on the type of [Sampler], this may produce a token id.
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> Result<&'a mut Logits>;

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
        res: &mut dyn HasSamplerResources,
        logits: &mut Logits,
    ) -> Result<Option<TID>> {
        let _ = self.sample(res, logits)?;
        Ok(self.sampled_token_id())
    }
}

impl Sampler for Box<dyn Sampler> {
    fn sampled_token_id(&self) -> Option<TID> {
        (**self).sampled_token_id()
    }

    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &mut Logits,
    ) -> Result<Option<TID>> {
        (**self).sample_token(res, logits)
    }

    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> Result<&'a mut Logits> {
        (**self).sample(res, logits)
    }
}

impl Sampler for Arc<Mutex<dyn Sampler>> {
    fn sampled_token_id(&self) -> Option<TID> {
        self.lock().ok()?.sampled_token_id()
    }

    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &mut Logits,
    ) -> Result<Option<TID>> {
        self.lock()
            .map_err(|e| SamplerError::InternalError(format!("Couldn't acquire lock: {e}")))?
            .sample_token(res, logits)
    }

    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> Result<&'a mut Logits> {
        self.lock()
            .map_err(|e| SamplerError::InternalError(format!("Couldn't acquire lock: {e}")))?
            .sample(res, logits)
    }
}
