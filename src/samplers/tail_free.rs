use crate::{configure::*, types::*};

/// # Tail free sampling
/// An approach to sampling that attempts to outperform existing
/// nucleus (top-p and top-k) methods.
/// See: <https://trentbrick.github.io/Tail-Free-Sampling/>
///
/// **Properties**:
/// - Modifies logits
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. Setting this to `0` is not recommended. (default: `1`)
/// - `z`: The z parameter. It is not entirely clear what a reasonable value here is but 1.0 appears to be
///   the same as disabled which is similar to top-p sampling. (default: `1.0`)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleTailFree {
    pub(crate) z: L,
    pub(crate) min_keep: usize,
}

impl Default for SampleTailFree {
    fn default() -> Self {
        Self {
            z: 1f32,
            min_keep: 1,
        }
    }
}

impl SampleTailFree {
    pub fn new(z: L, min_keep: usize) -> Self {
        Self { z, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn z(mut self, val: L) -> Self {
        self.z = val;
        self
    }
}

impl Sampler for SampleTailFree {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        use std::ops::ControlFlow::*;

        let Self { z, min_keep } = *self;

        if z >= 1f32 || logits.len() < 2 {
            return Ok(logits);
        }

        logits.ensure_softmax()?;

        let mut fderivs = logits
            .iter()
            .take(logits.len() - 1)
            .enumerate()
            .map(|(idx, l)| l.prob - logits[idx + 1].prob)
            .peekable();

        let want_sderivs = logits.len() - 2;
        let mut sderivs = Vec::with_capacity(want_sderivs);
        let mut ssum = 0f32;

        while let Some(prob) = fderivs.next() {
            let sprob = (prob
                - *fderivs.peek().ok_or_else(|| {
                    SamplerError::InternalError(String::from(
                        "Impossible: missing next deriv item?",
                    ))
                })?)
            .abs();
            ssum += sprob;
            sderivs.push(sprob);
            if sderivs.len() == want_sderivs {
                break;
            }
        }
        sderivs.iter_mut().for_each(|prob| *prob /= ssum);

        let mut cum_sum = 0f32;
        let last_idx =
            match sderivs
                .into_iter()
                .enumerate()
                .try_fold(logits.len(), |last_idx, (idx, prob)| {
                    cum_sum += prob;
                    if cum_sum > z && idx >= min_keep {
                        return Break(idx);
                    }
                    Continue(last_idx)
                }) {
                Continue(i) => i,
                Break(i) => i,
            };
        if last_idx != logits.len() {
            logits.truncate(last_idx);
            logits.set_softmax(false);
        }
        Ok(logits)
    }
}

impl ConfigurableSampler<usize, L> for SampleTailFree {}

impl HasSamplerMetadata<usize, L> for SampleTailFree {
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "tail free",
            description: Some(concat!(
                "An approach to sampling that attempts to ",
                "outperform existing nucleus (top-p and top-k) methods. ",
                "See: https://trentbrick.github.io/Tail-Free-Sampling/"
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "z",
                    description: Some(concat!(
                        "The z parameter. It is not entirely clear ",
                        "what a reasonable value here is but 1.0 appears to be the same ",
                        "as disabled which is similar to top-p sampling."
                    )),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "min_keep",
                    description: Some(concat!(
                        "Minimum number of tokens to keep after sampling. ",
                        "Setting this to 0 is not recommended."
                    )),
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
                    Some(SamplerOptionValueMut::Float(&mut self.z)),
                    Some(SamplerOptionValueMut::UInt(&mut self.min_keep)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.z)),
                    Some(SamplerOptionValue::UInt(self.min_keep)),
                ],
            )
        }
    }
}
