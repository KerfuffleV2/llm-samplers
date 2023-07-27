//! # LLM Samplers
//!
//! Rusty samplers for large language models.
//!
//! ## Examples
//!
//! You probably won't usually want to use individual [crate::types::Sampler]s. The most typical
//! use case is going to be chaining a number of samplers together. For reference, here's a link to
//! the [crate::samplers] module.
//!
//! A simple example of constructing a [crate::prelude::SamplerChain]:
//!
//! ```rust
//! use anyhow::Result;
//!
//! use llm_samplers::prelude::*;
//!
//! pub fn test_chain1() -> Result<()> {
//!
//!     let mut logits = Logits::try_from_iter([0.1f32, 0.2, 0.3, 0.4].into_iter())?;
//!
//!     // Demonstrating the different ways you can build a SamplerChain.
//!     // These are all equivalent.
//!     let mut sc = SamplerChain::new()
//!         + SampleFlatBias::new([(3, f32::NEG_INFINITY)]);
//!     sc += SampleTemperature::new(0.8);
//!     sc.push_sampler(SampleGreedy::new());
//!
//!     assert_eq!(
//!         sc.sample_token(
//!             // These samplers don't actually need any resources.
//!             &mut NilSamplerResources::default(),
//!             &mut logits)?,
//!         Some(1)
//!     );
//!
//!     // () also implements HasSamplerResources<TokenId = u32>
//!     // so you could use &mut () here.
//!     assert_eq!(sc.sample_token(&mut (), &mut logits)?, Some(1));
//!     Ok(())
//! }
//! ```
//!
//! Now let's look at something a bit more complicated:
//!
//! ```rust
//! # pub mod example {
//! use std::sync::{Arc, RwLock};
//!
//! use anyhow::Result;
//! use rand::{SeedableRng, rngs::StdRng};
//!
//! use llm_samplers::prelude::*;
//!
//! fn test_chain2() -> Result<()> {
//!     let example_logits = vec![0.1f32, 0.2, 0.3, 0.4];
//!     let mut res = SimpleSamplerResources::new(
//!         // Optionally include an RNG resource.
//!         Some(Box::new(StdRng::seed_from_u64(123))),
//!         // Optionally include a last tokens resource.
//!         Some(vec![]),
//!     );
//!     let mut logits = Logits::try_from_iter(example_logits.into_iter())?;
//!     let mut logits2 = logits.clone();
//!
//!     // SamplerChain with u32 token id type and f32 logit type.
//!     let mut sc = SamplerChain::<u32, f32>::new()
//!         // Bias logits (this example sets bias for token id 3 to -inf)
//!         + SampleFlatBias::new([(3, f32::NEG_INFINITY)])
//!         // Apply a repetition penalty.
//!         + SampleRepetition::new(1.1, 64)
//!         // Apply frequency and presence penalties.
//!         + SampleFreqPresence::new(0.05, 0.1, 64)
//!         // Apply temperature to logits.
//!         + SampleTemperature::new(0.8)
//!         // Sample a token using Mirostat1
//!         + SampleMirostat1::new(4, 5.0, 0.1);
//!
//!     // Put a value into `last_tokens`, this simulates us having already picked
//!     // that token (3) previously.
//!     res.with_last_tokens_mut(&mut |tokens| tokens.push(3u32))?;
//!
//!     assert_eq!(sc.sample_token(&mut res, &mut logits)?, Some(2));
//!
//!     // Now add the last selected token to the list.
//!     res.with_last_tokens_mut(&mut |tokens| tokens.push(2u32))?;
//!
//!     // And pick the next one. *Important*: Note that we don't reuse `logits`.
//!     // This is because `logits` already has all the filtering/sorting/permutation
//!     // from the previous sample call applied to it.
//!     assert_eq!(sc.sample_token(&mut res, &mut logits2)?, Some(1));
//!     Ok(())
//! }
//!
//! # }
//! ```
//!
//! ## Suggested chains/ordering
//! Suggestions based on the way `llama.cpp` does it.
//!
//! Note that you may not get meaningful results if you do weird stuff like chaining
//! multiple token selecting samplers. Based on available information, it also seems
//! like combining Mirostat samplers with top-K, top-P, etc will not work.
//!
//! ### Temperature sampling
//! 1. [SampleFlatBias](crate::samplers::SampleFlatBias) (optional)
//! 2. [SampleRepetition](crate::samplers::SampleRepetition) (optional)
//! 3. [SampleFreqPresence](crate::samplers::SampleFreqPresence) (optional)
//! 4. [SampleTopK](crate::samplers::SampleTopK) (optional)
//! 5. [SampleTailFree](crate::samplers::SampleTailFree) (optional)
//! 6. [SampleLocallyTypical](crate::samplers::SampleLocallyTypical) (optional)
//! 7. [SampleTopP](crate::samplers::SampleTopP) (optional)
//! 8. [SampleTemperature](crate::samplers::SampleTemperature) (optional)
//! 9. [SampleRandDistrib](crate::samplers::SampleRandDistrib)
//!
//! ### Mirostat V1/V2
//! 1. [SampleFlatBias](crate::samplers::SampleFlatBias) (optional)
//! 2. [SampleRepetition](crate::samplers::SampleRepetition) (optional)
//! 3. [SampleFreqPresence](crate::samplers::SampleFreqPresence) (optional)
//! 4. [SampleTemperature](crate::samplers::SampleTemperature) (optional)
//! 5. [SampleMirostat1](crate::samplers::SampleMirostat1) **or** [SampleMirostat2](crate::samplers::SampleMirostat2)

/// # Samplers live here!
///
/// Visiting the structs below is probably going to be more helpful than going to the modules.
pub mod samplers;

/// Types and traits.
pub mod types;

/// Sampler chains
mod chain;

/// Sampler resources
mod resource;

/// Configuring sampler options
pub mod configure;

#[cfg(test)]
mod tests;

/// Convenient rexports. The simplest way to use the crate is to just throw a
/// `use llm_samplers::prelude::*;`
/// into your project.
pub mod prelude {
    #[doc(inline)]
    pub use crate::{
        chain::*,
        configure::{ConfigurableSampler, ConfigureSamplerError},
        resource::*,
        samplers::*,
        types::*,
    };
}
