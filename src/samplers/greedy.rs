use crate::types::*;

/// # Greedy sampling
/// Selects the token with the highest logit value.
///
/// **Properties**:
/// - Selects a token
///
/// **Parameters**:
/// - (none)
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SampleGreedy<TID> {
    token_id: Option<TID>,
}

impl<TID: Clone> SampleGreedy<TID> {
    pub fn new() -> Self {
        Self { token_id: None }
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id.clone()
    }
}

impl<TID> std::ops::Deref for SampleGreedy<TID> {
    type Target = Option<TID>;

    fn deref(&self) -> &Self::Target {
        &self.token_id
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleGreedy<TID> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
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
