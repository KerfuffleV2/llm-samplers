use std::ops::{Add, AddAssign};

use anyhow::Result;
use thiserror::Error;

use crate::types::*;

use super::configurable::ConfigurableSampler;
use super::*;

#[derive(Debug, Error)]
pub enum BuildSamplersError {
    #[error("unknown slot name {0}")]
    UnknownSlot(String),

    #[error("cannot configure static slot {0}")]
    CannotConfigureStatic(String),

    #[error("configuring sampler {name} failed: {err}")]
    ConfigureFailed { name: String, err: anyhow::Error },
}

pub trait BuildableSampler<TID = u32, L = f32, UI = usize, F = f32>:
    Sampler<TID, L> + ConfigurableSampler<UI, F> + Send + Sync + std::fmt::Debug + 'static
where
    TID: CanTokenId,
    L: CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<T, TID, L, UI, F> BuildableSampler<TID, L, UI, F> for T
where
    Self: Sampler<TID, L> + ConfigurableSampler<UI, F> + Send + Sync + std::fmt::Debug + 'static,
    TID: CanTokenId,
    L: CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<TID, L, UI, F> Sampler<TID, L> for Box<dyn BuildableSampler<TID, L, UI, F>> {
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

pub type SamplerFactory<TID = u32, L = f32, UI = usize, F = f32> =
    dyn FnMut() -> Box<dyn BuildableSampler<TID, L, UI, F>>;

pub enum SamplerSlot<TID = u32, L = f32, UI = usize, F = f32> {
    /// Static slot holding a sampler that stays constant.
    Static {
        factory: Box<SamplerFactory<TID, L, UI, F>>,
    },

    /// A single optional sampler.
    Single {
        factory: Box<SamplerFactory<TID, L, UI, F>>,
        sampler: Option<Box<dyn BuildableSampler<TID, L, UI, F>>>,
    },

    /// A chain of samplers.
    Chain {
        factory: Box<SamplerFactory<TID, L, UI, F>>,
        samplers: Vec<Box<dyn BuildableSampler<TID, L, UI, F>>>,
    },
}

impl<TID, L, UI, F> SamplerSlot<TID, L, UI, F>
where
    TID: CanTokenId,
    L: CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    pub fn new_static(
        factory: impl FnMut() -> Box<dyn BuildableSampler<TID, L, UI, F>> + 'static,
    ) -> Self {
        Self::Static {
            factory: Box::new(factory),
        }
    }

    pub fn new_single(
        factory: impl FnMut() -> Box<dyn BuildableSampler<TID, L, UI, F>> + 'static,
        sampler: Option<impl BuildableSampler<TID, L, UI, F>>,
    ) -> Self {
        Self::Single {
            factory: Box::new(factory),
            sampler: sampler.map(|i| Box::new(i) as Box<dyn BuildableSampler<TID, L, UI, F>>),
        }
    }

    pub fn new_chain(
        factory: impl FnMut() -> Box<dyn BuildableSampler<TID, L, UI, F>> + 'static,
        samplers: impl IntoIterator<Item = Box<dyn BuildableSampler<TID, L, UI, F>>>,
    ) -> Self {
        Self::Chain {
            factory: Box::new(factory),
            samplers: samplers.into_iter().collect(),
        }
    }
}

pub struct SamplerChainBuilder<TID = u32, L = f32, UI = usize, F = f32> {
    slots: Vec<(String, SamplerSlot<TID, L, UI, F>)>,
}

impl<TID, L, UI, F> Default for SamplerChainBuilder<TID, L, UI, F> {
    fn default() -> Self {
        Self {
            slots: Default::default(),
        }
    }
}

impl<S: AsRef<str>, TID, L, UI, F, I: IntoIterator<Item = (S, SamplerSlot<TID, L, UI, F>)>> From<I>
    for SamplerChainBuilder<TID, L, UI, F>
{
    fn from(value: I) -> Self {
        Self {
            slots: value
                .into_iter()
                .map(|(name, slot)| (name.as_ref().to_string(), slot))
                .collect(),
        }
    }
}

impl<TID, L, UI, F> std::ops::Deref for SamplerChainBuilder<TID, L, UI, F> {
    type Target = Vec<(String, SamplerSlot<TID, L, UI, F>)>;

    fn deref(&self) -> &Self::Target {
        &self.slots
    }
}

impl<TID, L, UI, F> std::ops::DerefMut for SamplerChainBuilder<TID, L, UI, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slots
    }
}

impl<TID, L, UI, F> AddAssign<(String, SamplerSlot<TID, L, UI, F>)>
    for SamplerChainBuilder<TID, L, UI, F>
where
    TID: ConfigurableNumValue + CanTokenId,
    L: ConfigurableNumValue + CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    fn add_assign(&mut self, (name, slot): (String, SamplerSlot<TID, L, UI, F>)) {
        self.push_slot(name, slot)
    }
}

impl<TID, L, UI, F> Add<(String, SamplerSlot<TID, L, UI, F>)> for SamplerChainBuilder<TID, L, UI, F>
where
    TID: ConfigurableNumValue + CanTokenId,
    L: ConfigurableNumValue + CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    type Output = Self;

    fn add(mut self, (name, slot): (String, SamplerSlot<TID, L, UI, F>)) -> Self {
        self.push_slot(name, slot);
        self
    }
}

impl<TID, L, UI, F> SamplerChainBuilder<TID, L, UI, F>
where
    TID: ConfigurableNumValue + CanTokenId,
    L: ConfigurableNumValue + CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    pub fn push_slot(&mut self, name: String, slot: SamplerSlot<TID, L, UI, F>) {
        self.slots.push((name, slot))
    }

    pub fn configure(&mut self, name: impl AsRef<str>, s: impl AsRef<str>) -> Result<()> {
        let (name, s) = (name.as_ref(), s.as_ref());
        let cfgerr = |err| BuildSamplersError::ConfigureFailed {
            name: name.to_string(),
            err,
        };

        let (_, slot) = self
            .slots
            .iter_mut()
            .find(|(slotname, _slot)| slotname == name)
            .ok_or_else(|| BuildSamplersError::UnknownSlot(name.to_string()))?;

        match slot {
            SamplerSlot::Static { .. } => {
                Err(BuildSamplersError::CannotConfigureStatic(name.to_string()))?
            }
            SamplerSlot::Single { factory, sampler } => {
                if let Some(sampler) = sampler {
                    sampler.configure(s).map_err(cfgerr)?;
                } else {
                    let mut fresh = factory();
                    fresh.configure(s).map_err(cfgerr)?;
                    *sampler = Some(fresh);
                }
            }
            SamplerSlot::Chain { factory, samplers } => {
                let mut fresh = factory();
                fresh.configure(s).map_err(cfgerr)?;
                samplers.push(fresh);
            }
        }
        Ok(())
    }

    pub fn into_chain(self) -> SamplerChain<TID, L> {
        let mut chain = SamplerChain::<TID, L>::new();

        self.slots.into_iter().for_each(|(_name, slot)| match slot {
            SamplerSlot::Static { mut factory } => chain += factory(),
            SamplerSlot::Single { sampler, .. } => {
                if let Some(sampler) = sampler {
                    chain += sampler
                }
            }
            SamplerSlot::Chain { samplers, .. } => {
                samplers.into_iter().for_each(|sampler| chain += sampler)
            }
        });
        chain
    }
}
