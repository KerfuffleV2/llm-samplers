# llm-samplers

Token samplers for large language models, written in Rust!

## Status

Extremely early in development, poorly tested. You can look at [`src/tests.rs`](src/tests.rs) for some examples of use.

Also a fairly simple example of using Mirostat with my RWKV project here: https://github.com/KerfuffleV2/smolrsrwkv/blob/60b8e8bfe64f157f1800445128af3b4adbbc64c1/smolrwkv-cli/src/main.rs#L139-L164

## Samplers

Using the term "sampler" here loosely, perhaps it should be renamed in the future. Right now a "sampler"
could be something that manipulates the list of logits (for example, a top-k sampler might prune the list
to the top K entries), it might actually pick a token or both!

1. Flat bias - biases tokens by the specified amount
2. Frequency / presence - Applies frequency and presence penalties
3. Greedy - picks the token ID with the highest probability
4. Locally typical
5. Mirostat V1
6. Mirostat V2
7. Random distribution - picks a token ID based on weighted probabilities
8. Repetition - applies a repetition penalty
9. Tail free
10. Temperature
11. Top-K
12. Top-P

Real descriptions may (or may not happen) eventually. For now, you can check out the llama.cpp `main` example README for a brief overview of some of the types of sampler: https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md#generation-flags

## Example

 You probably won't usually want to use individual `Sampler`s. The most typical
 use case is going to be chaining a number of samplers together.

 A simple example of constructing a [SamplerChain]:

 ```rust
 use anyhow::Result;

 use llm_samplers::prelude::*;

 pub fn test_chain1() -> Result<()> {

     let mut logits = Logits::try_from_iter([0.1f32, 0.2, 0.3, 0.4].into_iter())?;

     // Demonstrating the different ways you can build a SamplerChain.
     // These are all equivalent.
     let mut sc = SamplerChain::new()
         + SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]);
     sc += SampleTemperature::new(0.8);
     sc.push_sampler(SampleGreedy::new());

     assert_eq!(
         sc.sample_token(
             // These samplers don't actually need any resources.
             &mut NilSamplerResources::default(),
             &mut logits)?,
         Some(1)
     );

     // () also implements HasSamplerResources<TokenId = u32>
     // so you could use &mut () here.
     assert_eq!(sc.sample_token(&mut (), &mut logits)?, Some(1));
     Ok(())
 }
 ```

 The previous example is simple but not very realistic: the greedy sampler doesn't
 even care about temperature. Now let's look at something a bit more complicated:

 ```rust
 use anyhow::Result;
 use rand::{SeedableRng, rngs::StdRng};

 use llm_samplers::prelude::*;

 fn test_chain2() -> Result<()> {
     let example_logits = vec![0.1f32, 0.2, 0.3, 0.4];
     let mut res = SimpleSamplerResources::new(
         // Optionally include an RNG resource.
         Some(Box::new(StdRng::seed_from_u64(123))),
         // Optionally include a last tokens resource.
         Some(vec![]),
     );
     let mut logits = Logits::try_from_iter(example_logits.into_iter())?;
     let mut logits2 = logits.clone();

     // SamplerChain with u32 token id type and f32 logit type.
     let mut sc = SamplerChain::<u32, f32>::new()
         // Bias logits (this example sets bias for token id 3 to -inf)
         + SampleFlatBias::new(&[(3, f32::NEG_INFINITY)])
         // Apply a repetition penalty.
         + SampleRepetition::new(1.1, 64)
         // Apply frequency and presence penalties.
         + SampleFreqPresence::new(0.05, 0.1, 64)
         // Apply temperature to logits.
         + SampleTemperature::new(0.8)
         // Sample a token using Mirostat1
         + SampleMirostat1::new(4, 5.0, 0.1);

     // Put a value into `last_tokens`, this simulates us having already picked
     // that token (3) previously.
     res.with_last_tokens_mut(&mut |tokens| tokens.push(3u32))?;

     assert_eq!(sc.sample_token(&mut res, &mut logits)?, Some(2));

     // Now add the last selected token to the list.
     res.with_last_tokens_mut(&mut |tokens| tokens.push(2u32))?;

     // And pick the next one. *Important*: Note that we don't reuse `logits`.
     // This is because `logits` already has all the filtering/sorting/permutation
     // from the previous sample call applied to it.
     assert_eq!(sc.sample_token(&mut res, &mut logits2)?, Some(1));
     Ok(())
 }
 ```

## Links

**Note**: Crate/docs version likely won't match this repo.

* Crate: https://crates.io/crates/llm-samplers
* Docs: https://docs.rs/llm-samplers/

## Credits

Initial version closely referenced from the samplers in the [llama.cpp](https://github.com/ggerganov/llama.cpp) project (although not
a line-by-line port). Thanks!
