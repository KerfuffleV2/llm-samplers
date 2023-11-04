use std::{collections::HashMap, hash::Hash, marker::PhantomData};

use crate::{configure::*, types::*};

/// # Sequence Repetition
/// This sampler penalizes repeating sequences of tokens that have already been seen within the
/// `last_n` window. It is fairly complicated, so here is an example. Suppose we have generated
/// this sequence of tokens: `1, 2, 3, 4, 1, 2, 3`
///
/// Disregarding `tolerance` and `max_merge` for now, if `min_length` is 3, then
/// `4` would be the token ID selected to penalize here. This is because the last
/// tokens are `1, 2, 3` and if we generate a `4` then have created a sequence that
/// already exists: `1, 2, 3, 4`.
///
/// If `tolerance` was `1` and the sequence was `1, 8, 3, 4, 1, 2, 3`
/// we would still penalize `4` since the `8` gets "tolerated". If we also set `max_merge=2` and
/// the sequence was `1, 7, 8, 3, 4, 1, 2, 3` it would _still_ be count as a match and `4` would
/// be penalized.
///
/// **Warning**: Very alpha code, likely has significant bugs.
///
/// **Properties**:
/// - Modifies logits
///
/// **Parameters**:
/// - `last_n`: Number of last tokens to consider. (default: `64`)
/// - `min_length`: The minimum length for a sequence to match. (default: `0`)
/// - `flat_penalty`: Flat penalty to apply to the token that would continue the matched sequence. (default: `0.0`)
/// - `stacking_penalty`: Stacking penalty to the token that would continue the matched sequence,
///     it is multiplied by the sequence length. (default: `0.0`)
/// - `tolerance`: Tolerance basically acts like a wildcard to allow fuzzy sequence matching.
///     For example, if tolerance is set to `1`, then `1, 6, 3` could match with `1, 2, 3`. (default: `0`)
/// - `max_merge`: Controls the number of consecutive non-matching tokens that
///     the tolerance wildcard can match. Setting this to `0` or `1` deactivates it.
///     Setting it to 2 would allow `1, 6, 6, 3` to match with `1, 2, 3`. (default: `1`)

#[derive(Debug, Clone)]
pub struct SampleSeqRepetition<TID = u32, L = f32> {
    flat_penalty: L,
    stacking_penalty: L,
    tolerance: usize,
    max_merge: usize,
    min_length: usize,
    last_n: usize,
    marker: PhantomData<TID>,
}

impl<TID: CanTokenId, L: CanLogit> Default for SampleSeqRepetition<TID, L> {
    fn default() -> Self {
        Self {
            flat_penalty: L::zero(),
            stacking_penalty: L::zero(),
            tolerance: 0,
            max_merge: 1,
            last_n: 64,
            min_length: 4,
            marker: PhantomData,
        }
    }
}

impl<TID: CanTokenId, L: CanLogit> SampleSeqRepetition<TID, L> {
    pub fn new(
        flat_penalty: L,
        stacking_penalty: L,
        min_length: usize,
        tolerance: usize,
        max_merge: usize,
        last_n: usize,
    ) -> Self {
        Self {
            flat_penalty,
            stacking_penalty,
            min_length,
            tolerance,
            max_merge,
            last_n,
            marker: PhantomData,
        }
    }

    pub fn min_length(mut self, val: usize) -> Self {
        self.min_length = val;
        self
    }

    pub fn tolerance(mut self, val: usize) -> Self {
        self.tolerance = val;
        self
    }

    pub fn max_merge(mut self, val: usize) -> Self {
        self.max_merge = val;
        self
    }

    pub fn last_n(mut self, val: usize) -> Self {
        self.last_n = val;
        self
    }

    pub fn flat_penalty(mut self, val: L) -> Self {
        self.flat_penalty = val;
        self
    }

    pub fn stacking_penalty(mut self, val: L) -> Self {
        self.stacking_penalty = val;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SeqMatchResult<'a, T> {
    h_offs: usize,
    h_len: usize,
    n_len: usize,
    seq: &'a [T],
}

fn find_seqs<T: PartialEq + Copy + std::fmt::Debug>(
    seq: &[T],
    min_len: usize,
    tolerance: usize,
    max_merge: usize,
) -> Vec<SeqMatchResult<'_, T>> {
    let seqlen = seq.len();
    if seqlen < min_len * 2 {
        return Vec::default();
    }

    let mut result = Vec::with_capacity(32);
    let mut hay = seq;

    while hay.len() > min_len {
        let mut nlen = min_len;
        let mut needle = &seq[seqlen - nlen..];

        while seqlen >= nlen + min_len {
            if hay[0] == needle[0] {
                fuzzy_match(hay, needle, needle.len(), tolerance, max_merge)
                    .into_iter()
                    .filter(|(hidx, _)| hay.len() > needle.len() && hay.len() > hidx + 1)
                    .for_each(|(hidx, mlen)| {
                        let mi = SeqMatchResult {
                            h_offs: seqlen - hay.len(),
                            h_len: hidx + 1,
                            n_len: mlen,
                            seq: &hay[..hidx + 1],
                        };
                        result.push(mi);
                    });
            }

            nlen += 1;
            if nlen >= hay.len() {
                break;
            }
            needle = &seq[seqlen - nlen..];
        }
        hay = &hay[1..];
    }
    result
}

fn fuzzy_match<T: PartialEq + std::fmt::Debug>(
    hay: &[T],
    needle: &[T],
    min_len: usize,
    mut tolerance: usize,
    merge_limit: usize,
) -> Vec<(usize, usize)> {
    let mut result = Vec::with_capacity(32);
    let mut window = 1;
    let mut hi = hay.iter().enumerate().fuse();

    'outer: for (nidx, n) in needle.iter().enumerate() {
        while window > 0 {
            window -= 1;
            let Some((hidx, h)) = hi.next() else {
                break 'outer;
            };
            if h == n {
                if nidx + 1 >= min_len {
                    result.push((hidx + 1, nidx + 1));
                }
                window = 1;
                continue 'outer;
            }
        }
        if tolerance == 0 {
            break;
        }
        tolerance -= 1;
        window = merge_limit + 1;
    }

    result
}

impl<TID: CanTokenId + Hash, L: CanLogit> Sampler<TID, L> for SampleSeqRepetition<TID, L> {
    fn sample<'a>(
        &mut self,
        res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let Self {
            last_n,
            min_length,
            flat_penalty,
            stacking_penalty,
            ..
        } = *self;

        if logits.is_empty()
            || (flat_penalty == L::zero() && stacking_penalty == L::zero())
            || min_length < 2
            || last_n < min_length
        {
            return Ok(logits);
        }

        let mut penalize: HashMap<TID, usize> = HashMap::with_capacity(64);

        res.with_last_tokens(&mut |orig_tokens| {
            if orig_tokens.len() < min_length * 2 {
                return;
            }
            let tokens = if last_n > orig_tokens.len() {
                orig_tokens
            } else {
                &orig_tokens[orig_tokens.len() - last_n..]
            };

            find_seqs(tokens, self.min_length, self.tolerance, self.max_merge)
                .into_iter()
                .filter(|mi| !mi.seq.is_empty())
                .for_each(|mi| {
                    let seqlen = mi.seq.len();
                    penalize
                        .entry(mi.seq[seqlen - 1])
                        .and_modify(|prevseqlen| *prevseqlen = (*prevseqlen).max(seqlen))
                        .or_insert(seqlen);
                });
        })?;

        for (tid, seqlen) in penalize.into_iter() {
            let tid = tid.to_usize().ok_or_else(|| {
                SamplerError::InternalError(String::from("TID conversion failed"))
            })?;
            if logits.len() <= tid {
                Err(SamplerError::InternalError(String::from(
                    "TID out of range for logits",
                )))?
            }
            let seqlen = L::from_usize(seqlen).ok_or_else(|| {
                SamplerError::InternalError(String::from("Couldn't convert usize to logit type"))
            })?;
            let l = &mut logits[tid].logit;

            *l = *l
                - (seqlen * stacking_penalty
                    + if seqlen > L::zero() {
                        L::one()
                    } else {
                        L::zero()
                    } * flat_penalty);
        }

        Ok(logits)
    }
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> ConfigurableSampler<usize, L>
    for SampleSeqRepetition<TID, L>
{
}

impl<TID: ConfigurableNumValue, L: ConfigurableNumValue> HasSamplerMetadata<usize, L>
    for SampleSeqRepetition<TID, L>
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "sequence repetition",
            description: Some(concat!(
                "Applies a penalty to tokens based on whether ",
                "they continue a sequence that was already seen"
            )),
            options: vec![
                SamplerOptionMetadata {
                    key: "flat_penalty",
                    description: Some(concat!(
                        "Flat penalty to apply to the token that ",
                        "would continue the matched sequence."
                    )),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "stacking_penalty",
                    description: Some(concat!(
                        "Stacking penalty to the token that would continue the matched sequence, ",
                        "it is multiplied by the sequence length."
                    )),
                    option_type: SamplerOptionType::Float,
                },
                SamplerOptionMetadata {
                    key: "min_length",
                    description: Some("The minimum length for a sequence to match."),
                    option_type: SamplerOptionType::UInt,
                },
                SamplerOptionMetadata {
                    key: "tolerance",
                    description: Some(concat!(
                        "Tolerance basically acts like a wildcard to ",
                        "allow fuzzy sequence matching. For example, if tolerance is set to 1, ",
                        "then [1, 6, 3] could match with [1, 2, 3]."
                    )),
                    option_type: SamplerOptionType::UInt,
                },
                SamplerOptionMetadata {
                    key: "max_merge",
                    description: Some(concat!(
                        "Controls the number of consecutive non-matching tokens that ",
                        "the tolerance wildcard can match. Setting this to 0 or 1 deactivates it. ",
                        "Setting it to 2 would allow [1, 6, 6, 3] to match with [1, 2, 3]."
                    )),
                    option_type: SamplerOptionType::UInt,
                },
                SamplerOptionMetadata {
                    key: ("last_n"),
                    description: Some(concat!(
                        "Number of previous tokens to consider when ",
                        "determining sequence repetition."
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
                    Some(SamplerOptionValueMut::Float(&mut self.flat_penalty)),
                    Some(SamplerOptionValueMut::Float(&mut self.stacking_penalty)),
                    Some(SamplerOptionValueMut::UInt(&mut self.min_length)),
                    Some(SamplerOptionValueMut::UInt(&mut self.tolerance)),
                    Some(SamplerOptionValueMut::UInt(&mut self.max_merge)),
                    Some(SamplerOptionValueMut::UInt(&mut self.last_n)),
                ],
            )
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, usize, L>> {
        unsafe {
            SamplerOptions::build_options(
                self.sampler_metadata().options,
                [
                    Some(SamplerOptionValue::Float(self.flat_penalty)),
                    Some(SamplerOptionValue::Float(self.stacking_penalty)),
                    Some(SamplerOptionValue::UInt(self.min_length)),
                    Some(SamplerOptionValue::UInt(self.tolerance)),
                    Some(SamplerOptionValue::UInt(self.max_merge)),
                    Some(SamplerOptionValue::UInt(self.last_n)),
                ],
            )
        }
    }
}
