use std::{
    fmt::Debug,
    ops::{Add, AddAssign},
};

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

pub trait BuildableSampler<UI, F>:
    Sampler + ConfigurableSampler<UI, F> + Send + Sync + std::fmt::Debug + 'static
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<T, UI, F> BuildableSampler<UI, F> for T
where
    Self: Sampler + ConfigurableSampler<UI, F> + Send + Sync + std::fmt::Debug + 'static,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<UI, F> Sampler for Box<dyn BuildableSampler<UI, F>> {
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

pub type SamplerFactory<UI = usize, F = f32> = dyn FnMut() -> Box<dyn BuildableSampler<UI, F>>;

pub enum SamplerSlot<UI, F> {
    /// Static slot holding a sampler that stays constant.
    Static { factory: Box<SamplerFactory<UI, F>> },

    /// A single optional sampler.
    Single {
        factory: Box<SamplerFactory<UI, F>>,
        sampler: Option<Box<dyn BuildableSampler<UI, F>>>,
    },

    /// A chain of samplers.
    Chain {
        factory: Box<SamplerFactory<UI, F>>,
        samplers: Vec<Box<dyn BuildableSampler<UI, F>>>,
    },
}

impl<UI: Debug, F: Debug> Debug for SamplerSlot<UI, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Static { .. } => f.debug_struct("Static").finish(),
            Self::Single { sampler, .. } => {
                f.debug_struct("Single").field("sampler", sampler).finish()
            }
            Self::Chain { samplers, .. } => {
                f.debug_struct("Chain").field("samplers", samplers).finish()
            }
        }
    }
}

impl<UI, F> SamplerSlot<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    pub fn new_static(factory: impl FnMut() -> Box<dyn BuildableSampler<UI, F>> + 'static) -> Self {
        Self::Static {
            factory: Box::new(factory),
        }
    }

    pub fn new_single(
        factory: impl FnMut() -> Box<dyn BuildableSampler<UI, F>> + 'static,
        sampler: Option<impl BuildableSampler<UI, F>>,
    ) -> Self {
        Self::Single {
            factory: Box::new(factory),
            sampler: sampler.map(|i| Box::new(i) as Box<dyn BuildableSampler<UI, F>>),
        }
    }

    pub fn new_chain(
        factory: impl FnMut() -> Box<dyn BuildableSampler<UI, F>> + 'static,
        samplers: impl IntoIterator<Item = Box<dyn BuildableSampler<UI, F>>>,
    ) -> Self {
        Self::Chain {
            factory: Box::new(factory),
            samplers: samplers.into_iter().collect(),
        }
    }

    /// For optional samplers (chain, single), this function will ensure
    /// the slot is populated with the default value. For chains this means
    /// it will add an item if the chain is empty. For single it will set
    /// the sampler to the default value if it's `None`.
    pub fn ensure_present(&mut self) {
        match self {
            SamplerSlot::Single { factory, sampler } if sampler.is_none() => {
                *sampler = Some(factory())
            }
            SamplerSlot::Chain { factory, samplers } if samplers.is_empty() => {
                samplers.push(factory())
            }
            _ => (),
        }
    }
}

#[derive(Debug)]
pub struct SamplerChainBuilder<UI, F> {
    slots: Vec<(String, SamplerSlot<UI, F>)>,
}

impl<UI, F> Default for SamplerChainBuilder<UI, F> {
    fn default() -> Self {
        Self {
            slots: Default::default(),
        }
    }
}

impl<S: AsRef<str>, UI, F, I: IntoIterator<Item = (S, SamplerSlot<UI, F>)>> From<I>
    for SamplerChainBuilder<UI, F>
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

impl<UI, F> std::ops::Deref for SamplerChainBuilder<UI, F> {
    type Target = Vec<(String, SamplerSlot<UI, F>)>;

    fn deref(&self) -> &Self::Target {
        &self.slots
    }
}

impl<UI, F> std::ops::DerefMut for SamplerChainBuilder<UI, F> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slots
    }
}

impl<UI, F> AddAssign<(String, SamplerSlot<UI, F>)> for SamplerChainBuilder<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    fn add_assign(&mut self, (name, slot): (String, SamplerSlot<UI, F>)) {
        self.push_slot(name, slot)
    }
}

impl<UI, F> Add<(String, SamplerSlot<UI, F>)> for SamplerChainBuilder<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    type Output = Self;

    fn add(mut self, (name, slot): (String, SamplerSlot<UI, F>)) -> Self {
        self.push_slot(name, slot);
        self
    }
}

impl<UI, F> SamplerChainBuilder<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    pub fn push_slot(&mut self, name: String, slot: SamplerSlot<UI, F>) {
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

    pub fn into_chain(self) -> SamplerChain {
        let mut chain = SamplerChain::new();

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
