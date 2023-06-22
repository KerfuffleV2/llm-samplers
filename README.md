# llm-samplers

Token samplers for large language models, written in Rust!

## Status

Extremely early in development, poorly tested. You can look at `src/tests.rs` for some examples of use.

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

## Credits

Initial version closely referenced from the samplers in the [llama.cpp](https://github.com/ggerganov/llama.cpp) project (although not
a line-by-line port). Thanks!
