use std::{borrow::Cow, str::FromStr};

use anyhow::Result;
use num_traits::{Float, FromPrimitive, NumCast};

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

/// Numeric values that can be used for configuring samplers.
pub trait ConfigurableNumValue: 'static + Copy + NumCast + FromPrimitive {}
impl<T> ConfigurableNumValue for T where T: 'static + Copy + NumCast + FromPrimitive {}

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
