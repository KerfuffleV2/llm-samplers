use std::fmt::Debug;

use rand::distributions::{uniform::SampleUniform, Distribution, WeightedIndex};

use crate::types::*;

/// # Random distribution sampling
/// A fancy way of saying the sampler selects a token
/// based on the probabilities. For example, if token X
/// has twice the probability value of token Y then we
/// can say token X will be twice as likely to be randomly
/// selected by this sampler.
///
/// **Properties**:
/// - Modifies logits
/// - Selects a token
///
/// **Parameters**:
/// - (none)
#[derive(Debug, Default)]
pub struct SampleRandDistrib<TID = u32> {
    token_id: Option<TID>,
}

impl<TID: CanTokenId> SampleRandDistrib<TID> {
    pub fn new() -> Self {
        Self { token_id: None }
    }
}

impl<TID, L> Sampler<TID, L> for SampleRandDistrib<TID>
where
    TID: CanTokenId,
    L: CanLogit + SampleUniform + Default + for<'a> std::ops::AddAssign<&'a L>,
{
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        self.token_id = None;
        if logits.is_empty() {
            return Ok(logits);
        }
        logits.softmax()?;
        let dist = WeightedIndex::new(logits.iter().map(|l| l.prob))
            .map_err(SamplerError::RandWeightedError)?;
        res.with_rng_mut(&mut |r| {
            self.token_id = Some(logits[dist.sample(r)].token_id);
        })?;
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

impl<UI, F> crate::configure::ConfigurableSampler<UI, F> for SampleRandDistrib<UI>
where
    UI: 'static + Copy + num_traits::NumCast + num_traits::FromPrimitive,
    F: 'static + Copy + num_traits::NumCast + num_traits::FromPrimitive,
{
}
