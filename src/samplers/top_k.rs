use crate::types::*;

/// # Top-K sampling
/// This sampler prunes all but the top `k` tokens in the list.
///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `k`: Number of entries to keep. (default: `40`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopK {
    k: usize,
    min_keep: usize,
}

impl Default for SampleTopK {
    fn default() -> Self {
        Self { min_keep: 1, k: 40 }
    }
}

impl SampleTopK {
    pub fn new(k: usize, min_keep: usize) -> Self {
        Self { k, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn k(mut self, val: usize) -> Self {
        self.k = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTopK {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
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
