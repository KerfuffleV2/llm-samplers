use num_traits::Float;
use rand::Rng;

use crate::{
    samplers::{rand_distrib::*, top_k::*},
    types::*,
};

/// Mirostat v2 sampling
pub struct SampleMirostat2<TID, L, R> {
    tau: L,
    eta: L,
    mu: L,
    token: Option<TID>,
    rd_sampler: RandDistribSampler<TID, R>,
}

impl<TID: CanTokenId, L: Float, R: Rng> SampleMirostat2<TID, L, R> {
    pub fn new<WR: WithRng<Rng = R, Output = usize> + 'static>(
        tau: L,
        eta: L,
        initial_mu: L,
        rng: Box<WR>,
    ) -> Self {
        Self {
            tau,
            eta,
            mu: initial_mu,
            rd_sampler: RandDistribSampler::<TID, R>::new(rng),
            token: None,
        }
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId, R: Rng> Sampler<TID, f32> for SampleMirostat2<TID, f32, R> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, f32>) -> &'a mut Logits<TID, f32> {
        self.token = None;
        if logits.is_empty() {
            return logits;
        }

        let Self { tau, eta, mu, .. } = *self;

        logits.softmax();
        let new_size = logits
            .iter()
            .enumerate()
            .find_map(|(idx, l)| (-l.prob.log2() > mu).then_some(idx))
            .unwrap_or_default()
            .max(1);
        logits.truncate(new_size);
        logits.softmax();
        self.rd_sampler.sample(logits);

        if let Some(tid) = self.rd_sampler.sample_token(logits) {
            let logit = logits
                .iter()
                .find(|l| l.token_id == tid)
                .expect("Impossible: sample token not in logits?");

            self.mu -= eta * (-logit.prob.log2() - tau);
            self.token = Some(tid);
        }
        logits
    }
}

impl<TID: CanTokenId, R: Rng> SampleToken<TID, f32> for SampleMirostat2<TID, f32, R> {
    fn sample_token(&mut self, logits: &mut Logits<TID, f32>) -> Option<TID> {
        self.sample(logits);
        self.token
    }
}

/// Mirostat v1 sampling
pub struct SampleMirostat1<TID, L, R> {
    n_vocab: usize,
    tau: L,
    eta: L,
    m: usize,
    mu: L,
    token: Option<TID>,
    rd_sampler: RandDistribSampler<TID, R>,
}

impl<TID: CanTokenId, L: Float, R: Rng> SampleMirostat1<TID, L, R> {
    pub fn new<WR: WithRng<Rng = R, Output = usize> + 'static>(
        n_vocab: usize,
        tau: L,
        eta: L,
        m: usize,
        initial_mu: L,
        rng: Box<WR>,
    ) -> Self {
        Self {
            n_vocab,
            tau,
            eta,
            m,
            mu: initial_mu,
            rd_sampler: RandDistribSampler::<TID, R>::new(rng),
            token: None,
        }
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId, R: Rng> Sampler<TID, f32> for SampleMirostat1<TID, f32, R> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, f32>) -> &'a mut Logits<TID, f32> {
        let Self {
            n_vocab,
            tau,
            eta,
            m,
            mu,
            ..
        } = *self;
        self.token = None;
        if logits.is_empty() || m < 1 {
            return logits;
        }
        let n_vocab = n_vocab as f32;

        logits.softmax();
        let (sum_ti_bi, sum_ti_sq) = logits
            .iter()
            .zip(logits.iter().skip(1))
            .enumerate()
            .take((m - 1).min(logits.len() - 1))
            .fold((0.0, 0.0), |(sum_ti_bi, sum_ti_sq), (idx, (l, l_next))| {
                let t_i = ((idx + 2) as f32 / (idx + 1) as f32).ln();
                let b_i = l.prob / l_next.prob;
                (sum_ti_bi + t_i * b_i, sum_ti_sq + t_i * t_i)
            });
        let s_hat = sum_ti_bi / sum_ti_sq;
        let epsilon_hat = s_hat - 1.0;
        let k = (epsilon_hat * mu.powf(2.0) / 1.0 - n_vocab.powf(-epsilon_hat)).powf(1.0 / s_hat)
            as usize;
        logits.sample(&mut SampleTopK::new(k, 1));

        if let Some(tid) = self.rd_sampler.sample_token(logits) {
            let logit = logits
                .iter()
                .find(|l| l.token_id == tid)
                .expect("Impossible: sample token not in logits?");

            self.mu -= eta * (-logit.prob.log2() - tau);
            self.token = Some(tid);
        }
        logits
    }
}

impl<TID: CanTokenId, R: Rng> SampleToken<TID, f32> for SampleMirostat1<TID, f32, R> {
    fn sample_token(&mut self, logits: &mut Logits<TID, f32>) -> Option<TID> {
        self.sample(logits);
        self.token
    }
}
