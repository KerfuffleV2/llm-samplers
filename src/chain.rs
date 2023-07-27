use std::{
    fmt::Debug,
    ops::{Add, AddAssign},
};

use crate::types::{CanLogit, CanTokenId, HasSamplerResources, Logits, Sampler};

#[derive(Default, Debug)]
/// A list of [Sampler]s that can be run in sequence. It implements `Sampler`
/// so you can build samplers as modular components. A typical use case would
/// be to have several filtering samplers and then a token-picking sampler as the last
/// item to enable calling [Sampler::sample_token] on the chain.
pub struct SamplerChain<TID = u32, L = f32> {
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
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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

impl<TID: CanTokenId, L: CanLogit, Rhs> AddAssign<Rhs> for SamplerChain<TID, L>
where
    Rhs: Sampler<TID, L> + Send + Sync + 'static,
{
    fn add_assign(&mut self, rhs: Rhs) {
        let _ = self.push_sampler(rhs);
    }
}

impl<TID: CanTokenId, L: CanLogit, Rhs> Add<Rhs> for SamplerChain<TID, L>
where
    Rhs: Sampler<TID, L> + Send + Sync + 'static,
{
    type Output = Self;

    fn add(mut self, rhs: Rhs) -> Self::Output {
        self += rhs;
        self
    }
}
