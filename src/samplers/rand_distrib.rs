use std::fmt::Debug;

use rand::distributions::{Distribution, WeightedIndex};

use crate::{configure::*, types::*};

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
#[derive(Debug, Default, Clone)]
pub struct SampleRandDistrib {
    token_id: Option<TID>,
}

impl SampleRandDistrib {
    pub fn new() -> Self {
        Self { token_id: None }
    }
}

impl Sampler for SampleRandDistrib {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        self.token_id = None;
        if logits.is_empty() {
            return Ok(logits);
        }
        logits.ensure_softmax()?;
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

impl<UI: ConfigurableNumValue, F: ConfigurableNumValue> ConfigurableSampler<UI, F>
    for SampleRandDistrib
{
}

impl<UI: ConfigurableNumValue, F: ConfigurableNumValue> HasSamplerMetadata<UI, F>
    for SampleRandDistrib
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "random distribution",
            description: Some("Randomly selects a token based on its probability."),
            options: vec![],
        }
    }
}
