//! # LLM Samplers
//!
//! Rusty samplers for large language models.
//!
//! ## Examples
//!
//! You probably won't usually want to use individual [crate::types::Sampler]. The most typical
//! use case is going to be chaining a number of samplers together: For reference, here's a link to
//! the [crate::samplers] module.
//!
//! ```
//! use anyhow::Result;
//!
//! use llm_samplers::prelude::*;
//!
//! pub fn test_chain1() -> Result<()> {
//!
//!     let mut logits = Logits::try_from_iter([0.1f32, 0.2, 0.3, 0.4].into_iter())?;
//!
//!     let mut sc = SamplerChain::new();
//!     sc.push_sampler(SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]))
//!         .push_sampler(SampleFlatBias::new(&[(2, f32::NEG_INFINITY)]))
//!         .push_sampler(SampleGreedy::new());
//!
//!     assert_eq!(sc.sample_token(&mut logits)?, Some(1));
//!     Ok(())
//! }
//! ```
//!
//! That example is simple but a bit unrealistic since you wouldn't normally
//! do weird stuff like chaining multiple flat bias samplers. Let's look at something
//! a bit more complicated.
//!
//! ```
//! # #[cfg(feature = "rand")]
//! # pub mod example {
//! use std::sync::{Arc, RwLock};
//!
//! use anyhow::Result;
//! use rand::rngs::StdRng;
//!
//! use llm_samplers::prelude::*;
//!
//! pub fn test_chain2() -> Result<()> {
//!     let testvals = vec![0.1f32, 0.2, 0.3, 0.4];
//!     let mut logits = Logits::try_from_iter(testvals.clone())?;
//!     let mut logits2 = logits.clone();
//!
//!     // This is an `Arc<RwLock<Vec>>` so the samplers that need it can hold a copy of the `Arc`
//!     // to read and we can also update the `Vec` in between sampling.
//!     let last_tokens = Arc::new(RwLock::new(vec![]));
//!
//!     let mut sc = SamplerChain::new();
//!     sc
//!         // Apply a repetition penalty.
//!         .push_sampler(SampleRepetition::new(1.1, 64, last_tokens.clone()))
//!
//!         // Apply frequency and presence penalities.
//!         .push_sampler(SampleFreqPresence::new(0.05, 0.1, 64, last_tokens.clone()))
//!
//!         // Apply temperature of 0.8 to the logits.
//!         .push_sampler(SampleTemperature::new(0.8))
//!
//!         // Bias token ID 3 to minus infinity. In other words: never select it.
//!         .push_sampler(SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]))
//!
//!         // Use a Mirostat v1 sampler to select a token at the end.
//!         .push_sampler(SampleMirostat1::<u32, f32, StdRng>::new(
//!             4,
//!             5.0,
//!             0.1,
//!             60,
//!             10.0,
//!             Box::new(RngBox::new_seedable(Some(123))),
//!         ));
//!
//!     // Put a value into `last_tokens`, this simulates us having already picked
//!     // that token (3) previously.
//!     last_tokens.write().unwrap().push(3);
//!     assert_eq!(sc.sample_token(&mut logits)?, Some(2));
//!
//!     // Now add the last selected token to the list.
//!     last_tokens.write().unwrap().push(2);
//!
//!     // And pick the next one. *Important*: Note that we don't reuse `logits`.
//!     // This is because `logits` already has all the filtering/sorting/permutation
//!     // from the previous sample call applied to it.
//!     assert_eq!(sc.sample_token(&mut logits2)?, Some(1));
//!     Ok(())
//! }
//! # }
//! ```

#[cfg(feature = "rand")]
/// Helper functions and types associated with random number generation.
pub mod rand;

/// Module containing the actual samplers.
pub mod samplers;

/// Types and traits.
pub mod types;

#[cfg(test)]
mod tests;

/// Convenient rexports. The simplest way to use the crate is to just throw a
/// `use llm_samplers::prelude::*;`
/// into your project.
pub mod prelude {
    #[cfg(feature = "rand")]
    #[doc(inline)]
    pub use crate::rand::*;

    #[doc(inline)]
    pub use crate::{samplers::*, types::*};
}
