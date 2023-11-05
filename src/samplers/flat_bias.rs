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
pub struct SampleFlatBias {
    pub(crate) bias: Vec<(TID, L)>,
}

impl std::ops::Deref for SampleFlatBias {
    type Target = Vec<(TID, L)>;

    fn deref(&self) -> &Self::Target {
        &self.bias
    }
}

impl std::ops::DerefMut for SampleFlatBias {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.bias
    }
}

impl SampleFlatBias {
    /// Construct the sampler from from anything that implements
    /// [IntoIterator] for the bias item type.
    pub fn new<I: IntoIterator<Item = (TID, L)>>(it: I) -> Self {
        Self {
            bias: Vec::from_iter(it),
        }
    }
}

impl Sampler for SampleFlatBias {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources,
        logits: &'a mut Logits,
    ) -> anyhow::Result<&'a mut Logits> {
        let valid_tid = 0..logits.len();
        self.bias.iter().for_each(|(tid, bv)| {
            let tid = *tid as usize;
            if valid_tid.contains(&tid) {
                let l = &mut logits[tid].logit;
                *l += *bv;
            }
        });
        Ok(logits)
    }
}

// FIXME: Find a sane way to implement this for the list of bias items.
impl<UI: ConfigurableNumValue, F: ConfigurableNumValue> ConfigurableSampler<UI, F>
    for SampleFlatBias
{
}

impl<UI: ConfigurableNumValue, F: ConfigurableNumValue> HasSamplerMetadata<UI, F>
    for SampleFlatBias
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
