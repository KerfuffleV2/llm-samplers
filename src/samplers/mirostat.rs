use std::fmt::Debug;

use num_traits::Float;

use crate::{
    samplers::{rand_distrib::*, top_k::*},
    types::*,
};

/// # Mirostat V1 sampling
///
/// *Note*: No [Default] instance since we can't know `n_vocab` in advance.
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
/// - Selects a token
///
/// **Parameters**:
/// - `n_vocab`: Model vocabulary size
/// - `eta`: Learning rate. (default: `0.1`)
/// - `tau`: Target entropy. (default: `5.0`)
/// - `m`: Unknown. Can be set manually after construction. (default: `100`)
/// - `mu`: Current learning state. Can be set manually after construction. (default: `tau * 2`)
#[derive(Debug)]
pub struct SampleMirostat1<TID, L> {
    n_vocab: usize,
    m: usize,
    tau: L,
    eta: L,
    mu: L,
    token: Option<TID>,
    rd_sampler: SampleRandDistrib<TID>,
}

impl<TID: CanTokenId, L: Float> SampleMirostat1<TID, L> {
    pub fn new(n_vocab: usize, tau: L, eta: L) -> Self {
        Self {
            n_vocab,
            tau,
            eta,
            m: 100,
            mu: tau * L::from(2.0f32).expect("Impossible: Can't convert f32 to Float"),
            rd_sampler: SampleRandDistrib::<TID>::new(),
            token: None,
        }
    }

    pub fn n_vocab(mut self, val: usize) -> Self {
        self.n_vocab = val;
        self
    }

    pub fn m(mut self, val: usize) -> Self {
        self.m = val;
        self
    }

    pub fn tau(mut self, val: L) -> Self {
        self.tau = val;
        self
    }

    pub fn eta(mut self, val: L) -> Self {
        self.eta = val;
        self
    }

    pub fn mu(mut self, val: L) -> Self {
        self.mu = val;
        self
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId> Sampler<TID, f32> for SampleMirostat1<TID, f32> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, f32>,
    ) -> Result<&'a mut Logits<TID, f32>, SamplerError> {
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
            return Ok(logits);
        }
        let n_vocab = n_vocab as f32;

        logits.softmax()?;
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
        logits.sample(res, &mut SampleTopK::new(k, 1))?;

        if let Some(tid) = self.rd_sampler.sample_token(res, logits)? {
            let logit = logits.iter().find(|l| l.token_id == tid).ok_or_else(|| {
                SamplerError::InternalError(String::from("Impossible: sample token not in logits?"))
            })?;

            self.mu -= eta * (-logit.prob.log2() - tau);
            self.token = Some(tid);
        }
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}

/// # Mirostat V2 sampling
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
/// - Selects a token
///
/// **Parameters**:
/// - `eta`: Learning rate. (default: `0.1`)
/// - `tau`: Target entropy. (default: `5.0`)
/// - `mu`: Current learning state. Can be set manually after construction. (default: `tau * 2`)
#[derive(Debug)]
pub struct SampleMirostat2<TID, L> {
    tau: L,
    eta: L,
    mu: L,
    token: Option<TID>,
    rd_sampler: SampleRandDistrib<TID>,
}

impl<TID: CanTokenId, L: Float> Default for SampleMirostat2<TID, L> {
    fn default() -> Self {
        Self {
            tau: L::from(5.0).expect("Impossible: Can't convert f32 to Float"),
            eta: L::from(0.1).expect("Impossible: Can't convert f32 to Float"),
            mu: L::from(10.0).expect("Impossible: Can't convert f32 to Float"),
            token: None,
            rd_sampler: SampleRandDistrib::<TID>::new(),
        }
    }
}

impl<TID: CanTokenId, L: Float> SampleMirostat2<TID, L> {
    pub fn new(tau: L, eta: L) -> Self {
        Self {
            tau,
            eta,
            mu: tau * L::from(2.0).expect("Impossible: Can't convert f32 to Float"),
            rd_sampler: SampleRandDistrib::<TID>::new(),
            token: None,
        }
    }

    pub fn tau(mut self, val: L) -> Self {
        self.tau = val;
        self
    }

    pub fn eta(mut self, val: L) -> Self {
        self.eta = val;
        self
    }

    pub fn mu(mut self, val: L) -> Self {
        self.mu = val;
        self
    }
}

// FIXME: Support logit types other than f32?
impl<TID: CanTokenId> Sampler<TID, f32> for SampleMirostat2<TID, f32> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, f32>,
    ) -> Result<&'a mut Logits<TID, f32>, SamplerError> {
        self.token = None;
        if logits.is_empty() {
            return Ok(logits);
        }

        let Self { tau, eta, mu, .. } = *self;

        logits.softmax()?;
        let new_size = logits
            .iter()
            .enumerate()
            .find_map(|(idx, l)| (-l.prob.log2() > mu).then_some(idx))
            .unwrap_or_default()
            .max(1);
        logits.truncate(new_size);
        logits.softmax()?;
        self.rd_sampler.sample(res, logits)?;

        if let Some(tid) = self.rd_sampler.sample_token(res, logits)? {
            let logit = logits.iter().find(|l| l.token_id == tid).ok_or_else(|| {
                SamplerError::InternalError(String::from("Impossible: sample token not in logits?"))
            })?;

            self.mu -= eta * (-logit.prob.log2() - tau);
            self.token = Some(tid);
        }
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}
