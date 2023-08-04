use num_traits::Float;

use crate::{configure::*, types::*};

/// # Flat bias sampling
/// Used to bias specific tokens by either increasing or decreasing their probability.
/// One common use case is to forbid certain tokens by setting them to negative infinity,
/// for example if you set the end of text token to `-inf` the LLM will keep generating
/// tokens.
///
/// This sampler implements [std::ops::Deref] and [std::ops::DerefMut] to the
/// internal [Vec] so you can freely manipulate the bias list.
///
/// **Properties**:
/// - Modifies logits
///
/// **Parameters**:
/// - `bias`: A [Vec] of token id and bias value tuples. (default: empty)
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SampleFlatBias<TID = u32, L = f32> {
    pub(crate) bias: Vec<(TID, L)>,
}

impl<TID, L> std::ops::Deref for SampleFlatBias<TID, L> {
    type Target = Vec<(TID, L)>;

    fn deref(&self) -> &Self::Target {
        &self.bias
    }
}

impl<TID, L> std::ops::DerefMut for SampleFlatBias<TID, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bias
    }
}

impl<TID: CanTokenId + 'static, L: Float + 'static> SampleFlatBias<TID, L> {
    /// Construct the sampler from from anything that implements
    /// [IntoIterator] for the bias item type.
    pub fn new<I: IntoIterator<Item = (TID, L)>>(it: I) -> Self {
        Self {
            bias: Vec::from_iter(it.into_iter()),
        }
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleFlatBias<TID, L> {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
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

// FIXME: Find a sane way to implement this for the list of bias items.
impl<
        TID: ConfigurableNumValue,
        L: ConfigurableNumValue,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    > ConfigurableSampler<UI, F> for SampleFlatBias<TID, L>
{
}

impl<
        TID: ConfigurableNumValue,
        L: ConfigurableNumValue,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    > HasSamplerMetadata<UI, F> for SampleFlatBias<TID, L>
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "sequence repetition",
            description: Some(concat!(
                "Used to bias specific tokens by either increasing or ",
                "decreasing their probability. ",
                "One common use case is to forbid certain tokens by ",
                "setting them to negative infinity,",
                "for example if you set the end of text token to `-inf` ",
                "the LLM will keep generating tokens."
            )),
            options: vec![],
        }
    }
}
