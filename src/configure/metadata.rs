use anyhow::Result;

use super::*;

/// Structure that defines a sampler option.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerOptionMetadata {
    /// Option name.
    pub key: &'static str,

    /// Optional option name.
    pub description: Option<&'static str>,

    /// The type of option.
    pub option_type: SamplerOptionType,
}

/// Structure that defines a sampler's metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerMetadata {
    pub name: &'static str,
    pub description: Option<&'static str>,
    pub options: Vec<SamplerOptionMetadata>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SamplerOptions<T>(Vec<(SamplerOptionMetadata, Option<T>)>);

impl<T> std::ops::Deref for SamplerOptions<T> {
    type Target = Vec<(SamplerOptionMetadata, Option<T>)>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for SamplerOptions<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Default for SamplerOptions<T> {
    fn default() -> Self {
        Self(Vec::default())
    }
}

impl<T, I: IntoIterator<Item = (SamplerOptionMetadata, Option<T>)>> From<I> for SamplerOptions<T> {
    fn from(value: I) -> Self {
        Self(value.into_iter().collect())
    }
}

impl<T> SamplerOptions<T> {
    /// Convenience function to build options from two iterators of
    /// equal length.
    ///
    /// ## Safety
    /// The metadata options in the first argument must match up with
    /// the second 1:1.
    pub unsafe fn build_options(
        md: impl IntoIterator<Item = SamplerOptionMetadata>,
        i: impl IntoIterator<Item = Option<T>>,
    ) -> Self {
        Self(md.into_iter().zip(i.into_iter()).collect())
    }

    pub fn find_option_definition(
        &self,
        key: &str,
    ) -> Result<(SamplerOptionMetadata, Option<usize>)> {
        let key = key.trim();
        let mut it = self.iter().enumerate().filter_map(|(idx, (omd, acc))| {
            omd.key
                .starts_with(key)
                .then(|| (omd.clone(), acc.is_some().then_some(idx)))
        });
        let Some((optdef, optidx)) = it.next() else {
            Err(ConfigureSamplerError::UnknownOrBadType(if key.is_empty() {
                        "<unspecified>".to_string()
                } else {
                    key.to_string()
                }))?
        };

        if it.next().is_some() {
            Err(ConfigureSamplerError::AmbiguousKey(key.to_string()))?
        }

        Ok((optdef.clone(), optidx))
    }
}

/// Configurable samplers will need to implement this trait. It provides
/// metadata for a sampler like its name, description as well as a list of
/// options and their types. It may also provide a way to directly access and
/// manipulate fields in the sampler's configuration. The built-in samplers
/// all implement this.
pub trait HasSamplerMetadata<UI = usize, F = f32>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    fn sampler_metadata(&self) -> SamplerMetadata {
        SamplerMetadata {
            name: "unknown",
            description: None,
            options: vec![],
        }
    }

    fn sampler_options(&self) -> SamplerOptions<SamplerOptionValue<'_, UI, F>> {
        SamplerOptions::default()
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, UI, F>> {
        SamplerOptions::default()
    }
}

// /// Convenience trait for samplers that supply metadata.
// ///
// /// This takes four type arguments because a sampler's option types
// /// don't necessarily have to match its tokenid/logit types.
// pub trait SamplerWithMetadata<TID = u32, L = f32, UI = usize, F = f32>:
//     Sampler<TID, L> + HasSamplerMetadata<UI, F>
// where
//     TID: CanTokenId,
//     F: CanLogit,
//     UI: ConfigurableNumValue,
//     F: ConfigurableNumValue,
// {
// }

// impl<T, TID, L, UI, F> SamplerWithMetadata<TID, L, UI, F> for T
// where
//     Self: Sampler<TID, L> + HasSamplerMetadata<UI, F>,
//     TID: CanTokenId,
//     F: CanLogit,
//     UI: ConfigurableNumValue,
//     F: ConfigurableNumValue,
// {
// }
