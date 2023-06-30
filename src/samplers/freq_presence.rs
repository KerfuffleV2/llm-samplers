use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, RwLock},
};

use crate::types::*;

/// Presence and frequency penalty sampling
#[derive(Debug, Clone)]
pub struct SampleFreqPresence<TID, L> {
    alpha_frequency: L,
    alpha_presence: L,
    last_n: usize,
    tokens: Arc<RwLock<Vec<TID>>>,
}

impl<TID: CanTokenId, L: CanLogit> SampleFreqPresence<TID, L> {
    pub fn new(
        alpha_frequency: L,
        alpha_presence: L,
        last_n: usize,
        tokens: Arc<RwLock<Vec<TID>>>,
    ) -> Self {
        Self {
            alpha_frequency,
            alpha_presence,
            last_n,
            tokens,
        }
    }
}

impl<TID: CanTokenId + Hash, L: CanLogit> Sampler<TID, L> for SampleFreqPresence<TID, L> {
    fn sample<'a>(
        &mut self,
        logits: &'a mut Logits<TID, L>,
    ) -> Result<&'a mut Logits<TID, L>, SamplerError> {
        let Self {
            alpha_frequency,
            alpha_presence,
            last_n,
            ..
        } = *self;
        let tokens = self.tokens.read().map_err(|e| {
            SamplerError::InternalError(format!("Couldn't acquire last tokens read lock: {e}"))
        })?;
        let tokens = if last_n > tokens.len() {
            &tokens[..]
        } else {
            &tokens[tokens.len() - last_n..]
        };
        let mut counts = HashMap::with_capacity(tokens.len());
        tokens.iter().for_each(|tid| {
            let cnt = counts.entry(tid).or_insert(L::zero());
            *cnt = *cnt + L::one()
        });

        logits.iter_mut().for_each(|l| {
            if let Some(cnt) = counts.get(&l.token_id) {
                l.logit = l.logit
                    - (*cnt * alpha_frequency
                        + if cnt > &L::zero() {
                            L::one()
                        } else {
                            L::zero()
                        } * alpha_presence);
            }
        });
        Ok(logits.set_sorted(false))
    }
}
