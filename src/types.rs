use std::ops::{Deref, DerefMut};

use num_traits::{Float, FromPrimitive, PrimInt, ToPrimitive};
use thiserror::Error;

#[derive(Debug, Error)]
/// Sampler errors
pub enum SamplerError {
    #[error("internal error: {0}")]
    /// General internal error type.
    InternalError(String),

    #[error("logits error: {0}")]
    /// Container for errors that occured while processing logits.
    LogitsError(LogitsError),

    #[cfg(feature = "rand")]
    #[error("rand error: {0}")]
    /// RNG-related errors
    RandError(rand::Error),

    #[cfg(feature = "rand")]
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
pub trait CanTokenId: PrimInt + FromPrimitive + ToPrimitive + Send + Sync {}

impl<T: PrimInt + FromPrimitive + ToPrimitive + Send + Sync> CanTokenId for T {}

/// Types that can be a logit implement this.
pub trait CanLogit: Float + Send + Sync {}

impl<T: Float + Send + Sync> CanLogit for T {}

#[derive(Debug, Clone, PartialEq)]
/// An individual logit with some additional metadata for use by the samplers.
pub struct Logit<TID, L> {
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
pub struct Logits<TID, L> {
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

impl<L: CanLogit> Logits<u32, L> {
    /// Make a new [Logits<u32, L>] from an iterator of `L`. We'd like to
    /// write this as [TryFrom] but unfortunately the types make this impossible.
    pub fn try_from_iter<I: IntoIterator<Item = L>>(it: I) -> Result<Self, LogitsError> {
        Ok(Self {
            sorted: false,
            logits: it
                .into_iter()
                .enumerate()
                .map(|(tid, logit)| {
                    if logit.is_nan() {
                        Err(LogitsError::InvalidLogit(tid))?
                    }
                    Ok(Logit {
                        token_id: tid as u32,
                        logit,
                        prob: L::zero(),
                    })
                })
                .collect::<Result<Vec<_>, LogitsError>>()?,
        })
    }
}

impl<L: CanLogit> TryFrom<Vec<L>> for Logits<u32, L> {
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
    pub fn ensure_sorted(&mut self) -> Result<&mut Self, LogitsError> {
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
    pub fn softmax(&mut self) -> Result<&mut Self, LogitsError> {
        if self.is_empty() {
            return Ok(self);
        }
        self.ensure_sorted()?;
        let max_l = self[0].logit;
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
        sampler: &mut S,
    ) -> Result<&mut Self, SamplerError> {
        sampler.sample(self)
    }

    /// Convenience method
    pub fn sample_token<S: Sampler<TID, L>>(
        &mut self,
        sampler: &mut S,
    ) -> Result<Option<TID>, SamplerError> {
        sampler.sample_token(self)
    }
}

/// The main sampler trait.
pub trait Sampler<TID, L>: Send + Sync {
    /// Runs the [Sampler]. Depending on the type of [Sampler], this may produce a token id.
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError>;

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
    fn sample_token(&mut self, logits: &mut Logits<TID, L>) -> Result<Option<TID>, SamplerError> {
        let _ = self.sample(logits)?;
        Ok(self.sampled_token_id())
    }
}

#[derive(Default)]
/// A list of [Sampler]s that can be run in sequence. It implements `Sampler`
/// so you can stick build samplers as modular components. A typical use case would
/// be to have several filtering samplers and then a token-picking sampler as the last
/// item to enable calling [Sampler::sample_token] on the chain.
pub struct SamplerChain<TID, L> {
    samplers: Vec<Box<dyn Sampler<TID, L>>>,
    token: Option<TID>,
}

impl<TID: CanTokenId, L: CanLogit> SamplerChain<TID, L> {
    pub fn new() -> Self {
        Self {
            samplers: vec![],
            token: None,
        }
    }

    // pub fn add_sampler_boxed(&mut self, sampler: Box<dyn Sampler<TID, L>>) -> &mut Self {
    //     self.token = None;
    //     self.samplers.push(sampler);
    //     self
    // }

    pub fn push_sampler(
        &mut self,
        sampler: impl Sampler<TID, L> + Send + Sync + 'static,
    ) -> &mut Self {
        self.token = None;
        self.samplers.push(Box::new(sampler));
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SamplerChain<TID, L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        self.token = None;
        self.samplers
            .iter_mut()
            .try_fold(logits, |logits, sampler| {
                let new_logits = sampler.sample(logits)?;
                self.token = sampler.sampled_token_id();
                Ok(new_logits)
            })
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}
