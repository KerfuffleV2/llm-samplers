use num_traits::Float;

use crate::types::*;

/// Flat bias sampling
#[derive(Debug, Clone, PartialEq)]
pub struct SampleFlatBias<'a, TID, L> {
    bias: &'a [(TID, L)],
}

impl<'a, TID: CanTokenId, L: Float> SampleFlatBias<'a, TID, L> {
    pub fn new(bias: &'a [(TID, L)]) -> Self {
        Self { bias }
    }
}

impl<'b, TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleFlatBias<'b, TID, L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let valid_tid = 0..logits.len();
        self.bias.iter().for_each(|(tid, bv)| {
            if let Some(tid) = tid.to_usize() {
                if valid_tid.contains(&tid) {
                    let l = &mut logits[tid].logit;
                    *l = *l + *bv;
                }
            }
        });
        Ok(logits)
    }
}
