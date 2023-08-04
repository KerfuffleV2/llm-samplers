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

mod build;
mod configurable;
mod metadata;
mod value;

use thiserror::Error;

#[doc(inline)]
pub use self::{build::*, configurable::*, metadata::*, value::*};

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
