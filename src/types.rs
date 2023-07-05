use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use num_traits::{Float, FromPrimitive, PrimInt, ToPrimitive};
use thiserror::Error;

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
pub trait CanTokenId:
    PrimInt + FromPrimitive + ToPrimitive + Debug + Clone + Copy + Send + Sync
{
}

impl<T: PrimInt + FromPrimitive + ToPrimitive + Debug + Clone + Copy + Send + Sync> CanTokenId
    for T
{
}

/// Types that can be a logit implement this.
pub trait CanLogit: Float + FromPrimitive + Debug + Clone + Send + Sync {}

impl<T: Float + FromPrimitive + Debug + Clone + Send + Sync> CanLogit for T {}

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
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        sampler: &mut S,
    ) -> Result<&mut Self, SamplerError> {
        sampler.sample(res, self)
    }

    /// Convenience method
    pub fn sample_token<S: Sampler<TID, L>>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        sampler: &mut S,
    ) -> Result<Option<TID>, SamplerError> {
        sampler.sample_token(res, self)
    }
}

/// The main sampler trait.
pub trait Sampler<TID, L>: Debug + Send + Sync {
    /// Runs the [Sampler]. Depending on the type of [Sampler], this may produce a token id.
    // fn sample<'a>(
    //     &mut self,
    //     logits: &'a mut Logits<TID, L>,
    // ) -> Result<&'a mut Logits<TID, L>, SamplerError>;

    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
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
    fn sample_token(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &mut Logits<TID, L>,
    ) -> Result<Option<TID>, SamplerError> {
        let _ = self.sample(res, logits)?;
        Ok(self.sampled_token_id())
    }
}

#[derive(Default, Debug)]
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
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        self.token = None;
        self.samplers
            .iter_mut()
            .try_fold(logits, |logits, sampler| {
                let new_logits = sampler.sample(res, logits)?;
                self.token = sampler.sampled_token_id();
                Ok(new_logits)
            })
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}

impl<TID: CanTokenId, L: CanLogit, Rhs> std::ops::AddAssign<Rhs> for SamplerChain<TID, L>
where
    Rhs: Sampler<TID, L> + Send + Sync + 'static,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let _ = self.push_sampler(rhs);
    }
}

impl<TID: CanTokenId, L: CanLogit, Rhs> std::ops::Add<Rhs> for SamplerChain<TID, L>
where
    Rhs: Sampler<TID, L> + Send + Sync + 'static,
{
    type Output = Self;

    fn add(mut self, rhs: Rhs) -> Self::Output {
        self += rhs;
        self
    }
}

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
