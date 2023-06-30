use std::sync::{Arc, RwLock};

use crate::types::*;

/// Repetition penalty sampling
#[derive(Debug, Clone)]
pub struct SampleRepetition<TID, L> {
    penalty: L,
    last_n: usize,
    tokens: Arc<RwLock<Vec<TID>>>,
}

impl<TID: CanTokenId, L: CanLogit> SampleRepetition<TID, L> {
    pub fn new(penalty: L, last_n: usize, tokens: Arc<RwLock<Vec<TID>>>) -> Self {
        Self {
            penalty,
            last_n,
            tokens,
        }
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleRepetition<TID, L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let Self {
            penalty, last_n, ..
        } = *self;
        let tokens = self.tokens.read().map_err(|e| {
            SamplerError::InternalError(format!("Couldn't acquire last tokens read lock: {e}"))
        })?;
        let tokens = if last_n > tokens.len() {
            &tokens[..]
        } else {
            &tokens[tokens.len() - last_n..]
        };
        logits
            .iter_mut()
            .filter(|l| tokens.contains(&l.token_id))
            .for_each(|l| {
                if l.logit <= L::zero() {
                    l.logit = l.logit * penalty
                } else {
                    l.logit = l.logit / penalty
                }
            });
        Ok(logits.set_sorted(false))
    }
}
