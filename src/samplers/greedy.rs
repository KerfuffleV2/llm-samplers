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
        if logits.is_empty() {
            self.token_id = None;
            return Ok(logits);
        }

        self.token_id = if logits.get_sorted() {
            logits.first()
        } else {
            logits
                .iter()
                .filter(|l| !l.logit.is_nan())
                .max_by(|x, y| x.logit.total_cmp(&y.logit))
        }
        .map(|l| l.token_id);

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
