#![deny(implied_bounds_entailment)]
use std::fmt::Debug;

use anyhow::Result;
use num_traits::{AsPrimitive, Float};
use rand::distributions::uniform::SampleUniform;

use crate::{
    configure::*,
    samplers::{rand_distrib::*, top_k::*},
    types::*,
};

/// # Mirostat V1 sampling
/// See: <https://arxiv.org/abs/2007.14966>
///
/// *Note*: The sampler does have a default implementation, however
/// it cannot be used until `n_vocab` is set.
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
#[derive(Debug, Clone)]
pub struct SampleMirostat1<TID = u32, L = f32> {
    pub(crate) n_vocab: usize,
    pub(crate) m: usize,
    pub(crate) tau: L,
    pub(crate) eta: L,
    pub(crate) mu: L,
    pub(crate) token: Option<TID>,
    rd_sampler: SampleRandDistrib<TID>,
}

impl<TID: CanTokenId, L: Float> Default for SampleMirostat1<TID, L> {
    fn default() -> Self {
        let five = L::one() + L::one() + L::one() + L::one() + L::one();
        let ten = five * (L::one() + L::one());

        Self {
            m: 100,
            eta: L::one() / ten,
            tau: five,
            mu: ten,
            token: None,
            rd_sampler: SampleRandDistrib::new(),
            n_vocab: 0,
        }
    }
}

impl<TID: CanTokenId, L: Float> SampleMirostat1<TID, L> {
    pub fn new(n_vocab: usize, tau: L, eta: L) -> Self {
        Self {
            n_vocab,
            tau,
            eta,
            m: 100,
            mu: tau * (L::one() + L::one()),
            rd_sampler: SampleRandDistrib::new(),
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

    /// Note: Setting tau will automatically set
    /// mu to `tau * 2`. If you need a custom
    /// value for mu, be sure to set it after tau.
    pub fn tau(mut self, val: L) -> Self {
        self.tau = val;
        self.mu = val * (L::one() + L::one());
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

impl<TID, L> Sampler<TID, L> for SampleMirostat1<TID, L>
where
    TID: CanTokenId,
    L: CanLogit + AsPrimitive<usize> + Default + SampleUniform + for<'a> std::ops::AddAssign<&'a L>,
{
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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
        if self.n_vocab == 0 {
            Err(SamplerError::InternalError(
                "Mirostat v1 sampler requires n_vocab".to_string(),
            ))?
        }
        let Some(n_vocab) = L::from(n_vocab) else {
            Err(LogitsError::InternalError(
                "Cannot convert n_vocab to sampler logits type".to_string(),
            ))?
        };
        let (zero, one, two) = (L::zero(), L::one(), L::one() + L::one());

        logits.softmax()?;
        let (sum_ti_bi, sum_ti_sq) = {
            let mut idx = zero;
            logits
                .iter()
                .zip(logits.iter().skip(1))
                .take((m - 1).min(logits.len() - 1))
                .fold((zero, zero), |(sum_ti_bi, sum_ti_sq), (l, l_next)| {
                    let t_i = ((idx + two) / (idx + one)).ln();
                    let b_i = l.prob / l_next.prob;
                    let result = (sum_ti_bi + t_i * b_i, sum_ti_sq + t_i * t_i);
                    idx = idx + one;
                    result
                })
        };
        let s_hat = sum_ti_bi / sum_ti_sq;
        let epsilon_hat = s_hat - one;
        let k = (epsilon_hat * mu.powf(two) / one - n_vocab.powf(-epsilon_hat)).powf(one / s_hat);
        logits.sample(res, &mut SampleTopK::new(k.as_(), 1))?;

        if let Some(tid) = self.rd_sampler.sample_token(res, logits)? {
            let logit = logits.iter().find(|l| l.token_id == tid).ok_or_else(|| {
                SamplerError::InternalError(String::from("Impossible: sample token not in logits?"))
            })?;

            self.mu = self.mu - (eta * (-logit.prob.log2() - tau));
            self.token = Some(tid);
        }
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue + Float> ConfigurableSampler<usize, L>
    for SampleMirostat1<TID, L>
{
    fn post_set_option(&mut self, md: &SamplerOptionMetadata) -> Result<()> {
        if md.key == "tau" {
            self.mu = self.tau * (L::one() + L::one());
        }
        Ok(())
    }
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> HasSamplerMetadata<usize, L>
    for SampleMirostat1<TID, L>
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "mirostat 1",
            description: Some("See: https://arxiv.org/abs/2007.14966"),
            options: vec![
                SamplerOptionMetadata {
                    key: "tau",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "eta",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "mu",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "m",
                    description: None,
                    option_type: SamplerOptionType::UInt,
                },
                SamplerOptionMetadata {
                    key: "n_vocab",
                    description: None,
                    option_type: SamplerOptionType::UInt,
                },
            ],
        }
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValueMut::Float(&mut self.tau)),
                    Some(SamplerOptionValueMut::Float(&mut self.eta)),
                    Some(SamplerOptionValueMut::Float(&mut self.mu)),
                    Some(SamplerOptionValueMut::UInt(&mut self.m)),
                    Some(SamplerOptionValueMut::UInt(&mut self.n_vocab)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.tau)),
                    Some(SamplerOptionValue::Float(self.eta)),
                    Some(SamplerOptionValue::Float(self.mu)),
                    Some(SamplerOptionValue::UInt(self.m)),
                    Some(SamplerOptionValue::UInt(self.n_vocab)),
                ],
            )
        }
    }
}

// *********************************************

/// # Mirostat V2 sampling
/// See: <https://arxiv.org/abs/2007.14966>
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
#[derive(Debug, Clone)]
pub struct SampleMirostat2<TID = u32, L = f32> {
    pub(crate) tau: L,
    pub(crate) eta: L,
    pub(crate) mu: L,
    pub(crate) token: Option<TID>,
    rd_sampler: SampleRandDistrib<TID>,
}

impl<TID: CanTokenId, L: Float> Default for SampleMirostat2<TID, L> {
    fn default() -> Self {
        let five = L::one() + L::one() + L::one() + L::one() + L::one();
        let ten = five * (L::one() + L::one());

        Self {
            eta: L::one() / ten,
            tau: five,
            mu: ten,
            token: None,
            rd_sampler: SampleRandDistrib::new(),
        }
    }
}

impl<TID: CanTokenId, L: Float> SampleMirostat2<TID, L> {
    pub fn new(tau: L, eta: L) -> Self {
        Self {
            tau,
            eta,
            mu: tau * (L::one() + L::one()),
            rd_sampler: SampleRandDistrib::new(),
            token: None,
        }
    }

    /// Note: Setting tau will automatically set
    /// mu to `tau * 2`. If you need a custom
    /// value for mu, be sure to set it after tau.
    pub fn tau(mut self, val: L) -> Self {
        self.tau = val;
        self.mu = val * (L::one() + L::one());
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

impl<TID, L> Sampler<TID, L> for SampleMirostat2<TID, L>
where
    TID: CanTokenId,
    L: CanLogit + Default + SampleUniform + for<'a> std::ops::AddAssign<&'a L>,
{
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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

            self.mu = self.mu - (eta * (-logit.prob.log2() - tau));
            self.token = Some(tid);
        }
        Ok(logits)
    }

    fn sampled_token_id(&self) -> Option<TID> {
        self.token
    }
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue + Float> ConfigurableSampler<usize, L>
    for SampleMirostat2<TID, L>
{
    fn post_set_option(&mut self, md: &SamplerOptionMetadata) -> Result<()> {
        if md.key == "tau" {
            self.mu = self.tau * (L::one() + L::one());
        }
        Ok(())
    }
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> HasSamplerMetadata<usize, L>
    for SampleMirostat2<TID, L>
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "mirostat 2",
            description: Some("See: https://arxiv.org/abs/2007.14966"),
            options: vec![
                SamplerOptionMetadata {
                    key: "tau",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "eta",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "mu",
                    description: None,
                    option_type: SamplerOptionType::Float,
                },
            ],
        }
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValueMut::Float(&mut self.tau)),
                    Some(SamplerOptionValueMut::Float(&mut self.eta)),
                    Some(SamplerOptionValueMut::Float(&mut self.mu)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.tau)),
                    Some(SamplerOptionValue::Float(self.eta)),
                    Some(SamplerOptionValue::Float(self.mu)),
                ],
            )
        }
    }
}
