use crate::types::*;

/// Top-K sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopK {
    k: usize,
    min_keep: usize,
}

impl SampleTopK {
    pub fn new(k: usize, min_keep: usize) -> Self {
        Self { k, min_keep }
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTopK {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let k = self.k.max(self.min_keep).min(logits.len());
        logits
            .ensure_sorted()
            .map_err(SamplerError::LogitsError)?
            .truncate(k);
        Ok(logits)
    }
}
