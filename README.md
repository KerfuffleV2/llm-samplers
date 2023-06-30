# llm-samplers

Token samplers for large language models, written in Rust!

## Status

Extremely early in development, poorly tested. You can look at [`src/tests.rs`](src/tests.rs) for some examples of use.

Also a very simple example of using Mirostat with my RWKV project here: https://github.com/KerfuffleV2/smolrsrwkv/blob/ce3cd93feac4ff3bf4ece0bcaf78ead262d8d57b/smolrwkv-cli/src/main.rs#L142-L176

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

 ```rust
use std::sync::{Arc, RwLock};

use anyhow::Result;
use rand::rngs::StdRng;

use llm_samplers::prelude::*;

pub fn test_chain() -> Result<()> {
    let testvals = vec![0.1f32, 0.2, 0.3, 0.4];
    let mut logits = Logits::try_from_iter(testvals.clone())?;
    let mut logits2 = logits.clone();

    // This is an `Arc<RwLock<Vec>>` so the samplers that need it can hold a copy of the `Arc`
    // to read and we can also update the `Vec` in between sampling.
    let last_tokens = Arc::new(RwLock::new(vec![]));

    let mut sc = SamplerChain::new();
    sc
        // Apply a repetition penalty.
        .push_sampler(SampleRepetition::new(1.1, 64, last_tokens.clone()))

        // Apply frequency and presence penalities.
        .push_sampler(SampleFreqPresence::new(0.05, 0.1, 64, last_tokens.clone()))

        // Apply temperature of 0.8 to the logits.
        .push_sampler(SampleTemperature::new(0.8))

        // Bias token ID 3 to minus infinity. In other words: never select it.
        .push_sampler(SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]))
        // Use a Mirostat v1 sampler to select a token at the end.
        .push_sampler(SampleMirostat1::<u32, f32, StdRng>::new(
            4,
            5.0,
            0.1,
            60,
            10.0,
            Box::new(RngBox::new_seedable(Some(123))),
        ));

    // Put a value into `last_tokens`, this simulates us having already picked
    // that token (3) previously.
    last_tokens.write().unwrap().push(3);
    assert_eq!(sc.sample_token(&mut logits)?, Some(2));

    // Now add the last selected token to the list.
    last_tokens.write().unwrap().push(2);

    // And pick the next one. *Important*: Note that we don't reuse `logits`.
    // This is because `logits` already has all the filtering/sorting/permutation
    // from the previous sample call applied to it.
    assert_eq!(sc.sample_token(&mut logits2)?, Some(1));
    Ok(())
}
 ```


## Credits

Initial version closely referenced from the samplers in the [llama.cpp](https://github.com/ggerganov/llama.cpp) project (although not
a line-by-line port). Thanks!
