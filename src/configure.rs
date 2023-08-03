//! This module implements configurable samplers
//! based on keys and values and also parsing strings
//! to configure a sampler.
//!
//! Unless you're writing your own implementation, you
//! basically only need to worry about
//! [ConfigurableSampler::configure](crate::configure::ConfigurableSampler::configure).
//! The built in samplers all implement this trait.
//!
//! Currently the default implementations aren't as flexible
//! as we might like and it should be noted that values are
//! converted from string to [u64] or [f64] before being
//! converted to the actual option type.

use std::{borrow::Cow, str::FromStr};

use anyhow::Result;
use num_traits::{Float, FromPrimitive, NumCast};
use thiserror::Error;

use crate::types::*;

/// Enum that holds the value for a sampler option.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerOptionValue<'a, UI = u64, F = f64> {
    /// Unsigned integer value.
    UInt(UI),

    /// Signed float value.
    Float(F),

    /// Boolean value.
    Bool(bool),

    /// String value.
    String(Cow<'a, str>),
}

/// Enum that holds a mutable reference to a sampler option value.
/// It's only necessary to worry about this when writing your own samplers and
/// implementing option parsing/handling.
#[derive(Debug, PartialEq)]
pub enum SamplerOptionValueMut<'a, UI, F> {
    /// Mutable reference to an unsigned integer value.
    UInt(&'a mut UI),

    /// Mutable reference to a signed float value.
    Float(&'a mut F),

    /// Mutable reference to a boolean value.
    Bool(&'a mut bool),

    /// Mutable reference to a string value.
    String(&'a mut Cow<'static, str>),
}

/// Sampler option types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerOptionType {
    /// Unsigned integer value.
    UInt,

    /// Signed float value.
    Float,

    /// Boolean value.
    Bool,

    /// String value.
    String,
}

/// Sampler option handling errors.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ConfigureSamplerError {
    #[error("unknown option key {0} or bad type")]
    /// Unknown option key or incorrect type specified.
    UnknownOrBadType(String),

    /// The supplied key can match multiple options.
    #[error("option key {0} is ambiguous")]
    AmbiguousKey(String),

    /// An error occurred converting the option value.
    #[error("option value conversion for key {0} failed")]
    ConversionFailure(String),

    /// The option value cannot be accessed as requested.
    #[error("option value for key {0} cannot be accessed as requested")]
    CannotAccessOptionValue(String),
}

impl<'a> SamplerOptionValue<'a> {
    /// Try to parse a string reference to an option value.
    ///
    /// Float options allow specifying `-inf` for negative infinity
    /// and `inf` or `+inf` for infinity.
    pub fn parse_value(typ: SamplerOptionType, s: impl AsRef<str>) -> Result<Self> {
        let s = s.as_ref();
        match typ {
            SamplerOptionType::UInt => Self::parse_uint(s).map(Self::UInt),
            SamplerOptionType::Float => Self::parse_float(s).map(Self::Float),
            SamplerOptionType::Bool => Self::parse_bool(s).map(Self::Bool),
            SamplerOptionType::String => Self::parse_string(s).map(Self::String),
        }
    }

    fn parse_uint(s: &str) -> Result<u64> {
        Ok(u64::from_str(s.trim())?)
    }

    fn parse_float(s: &str) -> Result<f64> {
        Ok(match s.trim() {
            "-inf" | "-INF" => f64::neg_infinity(),
            "inf" | "INF" | "+inf" | "+INF" => f64::infinity(),
            other => f64::from_str(other)?,
        })
    }

    fn parse_bool(s: &str) -> Result<bool> {
        Ok(match s.trim() {
            "true" | "t" | "yes" | "1" => true,
            "false" | "f" | "no" | "0" => false,
            _ => Err(SamplerError::InternalError(
                "Bad boolean value in sampler option".to_string(),
            ))?,
        })
    }

    fn parse_string(s: &str) -> Result<Cow<'static, str>> {
        Ok(Cow::Owned(s.trim().to_string()))
    }
}

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

/// Numeric values that can be used for configuring samplers.
pub trait ConfigurableNumValue: 'static + Copy + NumCast + FromPrimitive {}
impl<T> ConfigurableNumValue for T where T: 'static + Copy + NumCast + FromPrimitive {}

#[derive(Debug, Clone, PartialEq, Default)]
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
pub trait HasSamplerMetadata<UI = u32, F = f32>
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
        SamplerOptions(vec![])
    }

    fn sampler_options_mut(&mut self) -> SamplerOptions<SamplerOptionValueMut<'_, UI, F>> {
        SamplerOptions(vec![])
    }
}

/// Convenience trait for samplers that supply metadata.
///
/// This takes four type arguments because a sampler's option types
/// don't necessarily have to match its tokenid/logit types.
pub trait SamplerWithMetadata<TID = u32, L = f32, UI = u32, F = f32>:
    Sampler<TID, L> + HasSamplerMetadata<UI, F>
where
    TID: CanTokenId,
    F: CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

impl<T, TID, L, UI, F> SamplerWithMetadata<TID, L, UI, F> for T
where
    Self: Sampler<TID, L> + HasSamplerMetadata<UI, F>,
    TID: CanTokenId,
    F: CanLogit,
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
}

/// Configurable samplers implement this trait. "Configurable" means
/// they allow access to their their options by key/type and allow configuration
/// based on descriptions.
///
/// There are default implementations for for all the methods, so in the general
/// case you will only need to implement the `OPTIONS` constant with the option
/// definitions.
pub trait ConfigurableSampler<UI = u32, F = f32>: HasSamplerMetadata<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    /// Given an option key and [SamplerOptionValue] attempts to set the option
    /// to the specified value.
    ///
    /// The default implementation will call [Self::pre_set_option] before setting
    /// the option and [Self::post_set_option] afterward. For an example of
    /// the latter you can look at the Mirostat samplers: when `tau` is set, they'll
    /// automatically set `mu` to `tau * 2`.
    fn set_option(&mut self, key: &str, val: SamplerOptionValue) -> Result<()> {
        configurable_sampler::set_option(self, key, val)?;
        Ok(())
    }

    /// Called before an option is set and is passed a mutable reference
    /// to the [SamplerOptionValue]. It is also passed an index into
    /// the options definition list.
    ///
    /// The default implementation is a no-op.
    #[allow(unused_variables)]
    fn pre_set_option(
        &mut self,
        md: &SamplerOptionMetadata,
        val: &mut SamplerOptionValue,
    ) -> Result<()> {
        Ok(())
    }

    /// Called before an option is set is passed an index into
    /// the options definition list.
    ///
    /// The default implementation is a no-op.
    #[allow(unused_variables)]
    fn post_set_option(&mut self, md: &SamplerOptionMetadata) -> Result<()> {
        Ok(())
    }

    /// Gets an option by name.
    fn get_option(&self, key: &str) -> Result<SamplerOptionValue> {
        configurable_sampler::get_option(self, key)
    }

    /// Updates a sampler's configurable options based on a string in the
    /// format:
    ///
    /// `key1=value1:key2=value2:keyN=valueN`
    ///
    /// The key be a prefix of the option name as long as it's not
    /// ambiguous. It's also possible to just specify the value,
    /// which is equivalent to `=value` (i.e. a blank key name).
    ///
    /// Values in this default implementation cannot contain `=` or `:`
    /// and whitespace at the beginning and end of parts are stripped.
    fn configure(&mut self, s: &str) -> Result<()> {
        configurable_sampler::configure(self, s)?;
        Ok(())
    }
}

/// Since Rust traits don't allow calling base default methods from
/// a more specific implementation, the [ConfigurableSampler] trait
/// default methods are implemented in terms of the functions
/// in this submodule.
///
/// See the trait for descriptions of the methods.
pub mod configurable_sampler {
    use super::*;

    pub fn set_option<'a, CS, UI, F>(
        slf: &'a mut CS,
        key: &str,
        mut val: SamplerOptionValue,
    ) -> Result<&'a mut CS>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let key = key.trim();
        let (omd, optidx) = {
            let opts = slf.sampler_options_mut();
            if let (omd, Some(optidx)) = opts.find_option_definition(key)? {
                (omd, optidx)
            } else {
                Err(ConfigureSamplerError::CannotAccessOptionValue(
                    key.to_string(),
                ))?
            }
        };

        slf.pre_set_option(&omd, &mut val)?;

        let mut opts = slf.sampler_options_mut();
        let acc = opts[optidx]
            .1
            .take()
            .ok_or_else(|| ConfigureSamplerError::CannotAccessOptionValue(key.to_string()))?;

        match (acc, val) {
            (SamplerOptionValueMut::Float(rv), SamplerOptionValue::Float(v)) => {
                *rv = F::from_f64(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?
            }

            (SamplerOptionValueMut::UInt(rv), SamplerOptionValue::UInt(v)) => {
                *rv = UI::from_u64(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?
            }
            (SamplerOptionValueMut::Bool(rv), SamplerOptionValue::Bool(v)) => *rv = v,
            (SamplerOptionValueMut::String(rv), SamplerOptionValue::String(v)) => {
                *rv = Cow::from(v.to_string())
            }
            _ => Err(ConfigureSamplerError::UnknownOrBadType(key.to_string()))?,
        }
        slf.post_set_option(&omd)?;
        Ok(slf)
    }

    pub fn get_option<'a, CS, UI, F>(slf: &'a CS, key: &str) -> Result<SamplerOptionValue<'a>>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let key = key.trim();

        let mut opts = slf.sampler_options();

        let (_omd, Some(optidx)) = opts.find_option_definition(key)? else {
            Err(ConfigureSamplerError::CannotAccessOptionValue(key.to_string()))?
        };

        Ok(match opts[optidx].1.take().expect("Impossible") {
            SamplerOptionValue::UInt(v) => SamplerOptionValue::UInt(
                <u64 as NumCast>::from(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?,
            ),
            SamplerOptionValue::Float(v) => SamplerOptionValue::Float(
                <f64 as NumCast>::from(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?,
            ),
            SamplerOptionValue::Bool(v) => SamplerOptionValue::Bool(v),
            SamplerOptionValue::String(v) => SamplerOptionValue::String(Cow::from(v.to_string())),
        })
    }

    pub fn configure<CS, UI, F>(slf: &mut CS, s: &str) -> Result<()>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let opts = SamplerOptions(
            slf.sampler_options_mut()
                .iter()
                .map(|(md, acc)| (md.clone(), acc.is_some().then_some(())))
                .collect::<Vec<_>>(),
        );
        s.trim()
            .split(':')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .try_for_each(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or(("", kv));
                let (omd, Some(_)) = opts.find_option_definition(k)? else {
                    Err(ConfigureSamplerError::UnknownOrBadType(k.to_string()))?
                };

                slf.set_option(
                    omd.key,
                    SamplerOptionValue::parse_value(omd.option_type, v.trim())?,
                )?;
                anyhow::Ok(())
            })?;
        Ok(())
    }
}
