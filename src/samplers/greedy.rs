use num_traits::{Float, PrimInt};

pub use crate::types::*;

/// Greedy sampling
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SampleGreedy<TID> {
    token_id: Option<TID>,
}

impl<TID: Default + Clone> SampleGreedy<TID> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_token_id(&self) -> Option<TID> {
        self.token_id.clone()
    }
}

impl<TID: PrimInt, L: Float> Sampler<TID, L> for SampleGreedy<TID> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        self.token_id = None;
        if logits.is_empty() {
            return logits;
        }
        let mut result = logits[0].clone();
        logits.iter().skip(1).for_each(|l| {
            if l.logit > result.logit {
                result = l.clone()
            }
        });
        self.token_id = Some(result.token_id);
        logits
    }
}
