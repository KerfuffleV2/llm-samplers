use num_traits::{Float, PrimInt};

use crate::types::*;

/// Repetition penalty sampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRepetition<'a, TID, L> {
    penalty: L,
    last_n: usize,
    tokens: &'a [TID],
}

impl<'a, TID: PrimInt, L: Float> SampleRepetition<'a, TID, L> {
    pub fn new(penalty: L, last_n: usize, tokens: &'a [TID]) -> Self {
        Self {
            penalty,
            last_n,
            tokens,
        }
    }

    pub fn set_tokens(&mut self, tokens: &'a [TID]) -> &mut Self {
        self.tokens = tokens;
        self
    }
}

impl<'slf, TID: PrimInt, L: Float> Sampler<TID, L> for SampleRepetition<'slf, TID, L> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L> {
        let Self {
            penalty,
            last_n,
            tokens,
        } = *self;
        let tokens = if last_n > tokens.len() {
            tokens
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
        logits.set_sorted(false)
    }
}
