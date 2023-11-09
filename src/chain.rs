use std::{
    fmt::Debug,
    ops::{Add, AddAssign},
};

use crate::types::{HasSamplerResources, Logits, Sampler, TID};

#[derive(Default, Debug)]
/// A list of [Sampler]s that can be run in sequence. It implements `Sampler`
/// so you can build samplers as modular components. A typical use case would
/// be to have several filtering samplers and then a token-picking sampler as the last
/// item to enable calling [Sampler::sample_token] on the chain.
pub struct SamplerChain {
    samplers: Vec<Box<dyn Sampler>>,
    token: Option<TID>,
}

impl SamplerChain {
    pub fn new() -> Self {
        Self {
            samplers: vec![],
            token: None,
        }
    }

    pub fn push_sampler(&mut self, sampler: impl Sampler + Send + Sync + 'static) -> &mut Self {
        self.token = None;
        self.samplers.push(Box::new(sampler));
        self
    }
}

impl Sampler for SamplerChain {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
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

impl<Rhs> AddAssign<Rhs> for SamplerChain
where
    Rhs: Sampler + Send + Sync + 'static,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let _ = self.push_sampler(rhs);
    }
}

impl<Rhs> Add<Rhs> for SamplerChain
where
    Rhs: Sampler + Send + Sync + 'static,
{
    type Output = Self;

    fn add(mut self, rhs: Rhs) -> Self::Output {
        self += rhs;
        self
    }
}
