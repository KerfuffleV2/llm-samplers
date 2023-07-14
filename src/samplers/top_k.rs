use crate::{configure::*, types::*};

/// # Top-K sampling
/// This sampler prunes all but the top `k` tokens in the list.
///
/// **Properties**:
/// - Filters logits
///
/// **Parameters**:
/// - `min_keep`: Minimum number of entries to keep. (default: `1`)
/// - `k`: Number of entries to keep. (default: `40`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleTopK {
    pub(crate) k: usize,
    pub(crate) min_keep: usize,
}

impl Default for SampleTopK {
    fn default() -> Self {
        Self { min_keep: 1, k: 40 }
    }
}

impl SampleTopK {
    pub fn new(k: usize, min_keep: usize) -> Self {
        Self { k, min_keep }
    }

    pub fn min_keep(mut self, val: usize) -> Self {
        self.min_keep = val;
        self
    }

    pub fn k(mut self, val: usize) -> Self {
        self.k = val;
        self
    }
}

impl<TID: CanTokenId, L: CanLogit> Sampler<TID, L> for SampleTopK {
    fn sample<'a>(
        &mut self,
        _res: &mut dyn HasSamplerResources<TokenId = TID>,
        logits: &'a mut Logits<TID, L>,
    ) -> anyhow::Result<&'a mut Logits<TID, L>> {
        let k = self.k.max(self.min_keep).min(logits.len());
        logits.ensure_sorted()?.truncate(k);
        Ok(logits)
    }
}

impl<L> ConfigurableSampler<usize, L> for SampleTopK
where
    L: CanLogit + 'static,
{
    const OPTIONS: &'static [SamplerOptionDefinition<Self, usize, L>] = &[
        SamplerOptionDefinition {
            key: "k",
            desc: None,
            typ: SamplerOptionType::UInt,
            get: |slf| SamplerOptionValue::UInt(slf.k),
            get_mut: |slf| SamplerOptionValueMut::UInt(&mut slf.k),
        },
        SamplerOptionDefinition {
            key: "min_keep",
            desc: None,
            typ: SamplerOptionType::UInt,
            get: |slf| SamplerOptionValue::UInt(slf.min_keep),
            get_mut: |slf| SamplerOptionValueMut::UInt(&mut slf.min_keep),
        },
    ];
}
