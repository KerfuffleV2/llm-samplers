use std::fmt::Debug;

use rand::distributions::{Distribution, WeightedIndex};

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
pub struct SampleRandDistrib<TID> {
    token_id: Option<TID>,
}

impl<TID: CanTokenId> SampleRandDistrib<TID> {
    pub fn new() -> Self {
        Self { token_id: None }
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId> Sampler<TID, f32> for SampleRandDistrib<TID> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, f32>,
    ) -> Result<&'a mut Logits<TID, f32>, SamplerError> {
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
        self.get_token_id()
    }
}
