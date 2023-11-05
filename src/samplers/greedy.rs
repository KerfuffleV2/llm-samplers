use crate::{configure::*, types::*};

/// # Greedy sampling
/// Selects the token with the highest logit value.
///
/// **Properties**:
/// - Selects a token
///
/// **Parameters**:
/// - (none)
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SampleGreedy {
    token_id: Option<TID>,
}

impl SampleGreedy {
    pub fn new() -> Self {
        Self { token_id: None }
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

impl std::ops::Deref for SampleGreedy {
    type Target = Option<TID>;

    fn deref(&self) -> &Self::Target {
        &self.token_id
    }
}

impl Sampler for SampleGreedy {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        self.token_id = None;
        if logits.is_empty() {
            return Ok(logits);
        }
        let mut result = logits[0].clone();
        logits.iter().skip(1).for_each(|l| {
            if l.logit > result.logit {
                result = l.clone()
            }
        });
        self.token_id = Some(result.token_id);
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token_id
    }
}

impl<UI, F> ConfigurableSampler<UI, F> for SampleGreedy
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<UI, F> HasSamplerMetadata<UI, F> for SampleGreedy
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "greedy",
            description: Some("Selects the token with the highest logit value."),
            options: vec![],
        }
    }
}
