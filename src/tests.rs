use std::sync::{Arc, RwLock};

use anyhow::Result;
#[cfg(feature = "rand")]
use rand::rngs::StdRng;

use crate::prelude::*;

pub const T1: &[f32] = &[0.1, 0.2, 0.3, 0.4];
pub const TE1: &[f32] = &[0.4, 0.3, 0.2, 0.1];

pub type TestValidator<S> = fn(&mut S, &mut Logits<u32, f32>, &[f32]);

fn test_sampler_ll<S: Sampler<u32, f32>>(
    use_ln: bool,
    use_sm: bool,
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    let mut logits = Logits::try_from_iter(input.iter().map(|i| if use_ln { i.ln() } else { *i }))
        .expect("Bad logits");
    if use_sm {
        logits.softmax().expect("Softmax failed");
    }
    let result_logits = sampler.sample(&mut logits).expect("Sampler error");
    vf(sampler, result_logits, expected)
}

fn test_sampler<S: Sampler<u32, f32>>(
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(true, true, sampler, input, expected, vf)
}

fn test_sampler_no_sm<S: Sampler<u32, f32>>(
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(true, false, sampler, input, expected, vf)
}

fn test_sampler_raw<S: Sampler<u32, f32>>(
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(false, false, sampler, input, expected, vf)
}

fn validate(
    _sampler: &mut impl Sampler<u32, f32>,
    logits: &mut Logits<u32, f32>,
    expected: &[f32],
) {
    let result = logits
        .iter()
        .zip(expected.iter())
        .map(|(l, e)| (l.prob - e).abs())
        .collect::<Vec<_>>();
    // println!("initial:\n{logits:?}\nexpected:\n{expected:?}\ngot:\n{result:?}");
    assert_eq!(result.len(), expected.len());
    assert!(result.into_iter().all(|i| i < 0.00001))
}

fn validate_sm(
    sampler: &mut impl Sampler<u32, f32>,
    logits: &mut Logits<u32, f32>,
    expected: &[f32],
) {
    validate(sampler, logits.softmax().expect("Softmax failed"), expected);
}

fn validate_eq(
    _sampler: &mut impl Sampler<u32, f32>,
    logits: &mut Logits<u32, f32>,
    expected: &[f32],
) {
    assert_eq!(logits.iter().map(|l| l.logit).collect::<Vec<_>>(), expected)
}

fn do_test_greedy(it: impl Iterator<Item = f32>, expected: Option<u32>) -> Result<()> {
    assert_eq!(
        Logits::try_from_iter(it)?.sample_token(&mut SampleGreedy::new())?,
        expected
    );
    Ok(())
}

#[test]
fn test_greedy() -> Result<()> {
    do_test_greedy(T1.iter().copied(), Some(3))?;
    do_test_greedy(T1.iter().rev().copied(), Some(0))
}

#[test]
fn test_top_k() {
    test_sampler(&mut SampleTopK::new(1, 0), T1, &TE1[0..1], validate);
    test_sampler(&mut SampleTopK::new(3, 0), T1, &TE1[0..3], validate);
}

#[test]
fn test_top_p() {
    test_sampler(&mut SampleTopP::new(0.0, 1), T1, &TE1[0..1], validate);
    test_sampler(&mut SampleTopP::new(0.7, 1), T1, &TE1[0..2], validate);
    test_sampler(&mut SampleTopP::new(1.0, 1), T1, TE1, validate);
}

#[test]
fn test_repetition() {
    const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];
    const TER2: &[f32] = &[0.5, 0.5, 0.0, 0.0, 0.0];

    test_sampler(
        &mut SampleRepetition::new(50.0, 100, Arc::new(RwLock::new(vec![0]))),
        T,
        &[0.25, 0.25, 0.25, 0.25, 0.0],
        validate_sm,
    );
    test_sampler(
        &mut SampleRepetition::new(50.0, 100, Arc::new(RwLock::new(vec![0, 1, 2]))),
        T,
        TER2,
        validate_sm,
    );
    test_sampler(
        &mut SampleRepetition::new(50.0, 100, Arc::new(RwLock::new(vec![0, 1, 2, 0, 0]))),
        T,
        TER2,
        validate_sm,
    );
}

#[test]
fn test_freq_presence() {
    const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];

    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, 100, Arc::new(RwLock::new(vec![0]))),
        T,
        &[0.249997, 0.249997, 0.249997, 0.249997, 0.000011],
        validate_sm,
    );
    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, 100, Arc::new(RwLock::new(vec![0, 1, 2]))),
        T,
        &[0.499966, 0.499966, 0.000023, 0.000023, 0.000023],
        validate_sm,
    );
    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, 100, Arc::new(RwLock::new(vec![0, 1, 2, 0, 0]))),
        T,
        &[0.499977, 0.499977, 0.000023, 0.000023, 0.0],
        validate_sm,
    );
}

#[test]
fn test_typical() {
    test_sampler_no_sm(
        &mut SampleTypical::new(0.5, 1),
        &[0.97, 0.01, 0.01, 0.01],
        &[0.97],
        validate,
    );
    test_sampler_no_sm(
        &mut SampleTypical::new(0.5, 1),
        &[0.4, 0.2, 0.2, 0.2],
        &[0.2, 0.2, 0.2],
        validate,
    );
}

#[test]
fn test_tail_free() {
    const T: &[f32] = &[0.1, 0.15, 0.2, 0.25, 0.3];

    test_sampler_no_sm(&mut SampleTailFree::new(0.25, 1), T, &[0.3], validate);
    test_sampler_no_sm(&mut SampleTailFree::new(0.75, 1), T, &[0.3, 0.25], validate);
    test_sampler_no_sm(&mut SampleTailFree::new(0.99, 1), T, &[0.3, 0.25], validate);
}

#[test]
fn test_flat_bias() {
    const T: &[f32] = &[0.1, 0.15, 0.2, 0.25, 0.3];

    test_sampler_raw(
        &mut SampleFlatBias::new(&[(0, f32::NEG_INFINITY)]),
        T,
        &[f32::NEG_INFINITY, 0.15, 0.2, 0.25, 0.3],
        validate_eq,
    );
    test_sampler_raw(
        &mut SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]),
        T,
        &[0.1, 0.15, 0.2, f32::NEG_INFINITY, 0.3],
        validate_eq,
    );
}

#[test]
#[cfg(feature = "rand")]
fn test_rand_distrib() -> Result<()> {
    let mut sampler =
        RandDistribSampler::<u32, StdRng>::new(Box::new(RngBox::new_seedable(Some(123))));
    assert_eq!(
        Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(0)
    );
    assert_eq!(
        Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(2)
    );
    Ok(())
}

#[test]
#[cfg(feature = "rand")]
fn test_mirostat1() -> Result<()> {
    let mut sampler = SampleMirostat1::<u32, f32, StdRng>::new(
        3,
        5.0,
        0.1,
        100,
        10.0,
        Box::new(RngBox::new_seedable(Some(123))),
    );
    assert_eq!(
        Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(0)
    );
    let mut sampler = SampleMirostat1::<u32, f32, StdRng>::new(
        3,
        5.0,
        0.1,
        100,
        10.0,
        Box::new(RngBox::new_seedable(Some(123))),
    );
    assert_eq!(
        Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(2)
    );
    Ok(())
}

#[test]
#[cfg(feature = "rand")]
fn test_mirostat2() -> Result<()> {
    let mut sampler = SampleMirostat2::<u32, f32, StdRng>::new(
        5.0,
        0.1,
        10.0,
        Box::new(RngBox::new_seedable(Some(123))),
    );
    assert_eq!(
        Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(0)
    );
    let mut sampler = SampleMirostat2::<u32, f32, StdRng>::new(
        5.0,
        0.1,
        10.0,
        Box::new(RngBox::new_seedable(Some(123))),
    );
    assert_eq!(
        Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
            .sample_token(&mut sampler)?,
        Some(2)
    );
    Ok(())
}

#[test]
fn test_chain1() -> anyhow::Result<()> {
    let mut logits = Logits::try_from_iter(T1.iter().copied())?;

    let mut sc = SamplerChain::new();
    sc.push_sampler(SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]))
        .push_sampler(SampleFlatBias::new(&[(2, f32::NEG_INFINITY)]))
        .push_sampler(SampleGreedy::new());

    assert_eq!(sc.sample_token(&mut logits)?, Some(1));
    Ok(())
}

#[test]
#[cfg(feature = "rand")]
fn test_chain2() -> Result<()> {
    let mut logits = Logits::try_from_iter(T1.iter().copied())?;
    let mut logits2 = logits.clone();
    let last_tokens = Arc::new(RwLock::new(vec![]));

    let mut sc = SamplerChain::new();
    sc.push_sampler(SampleRepetition::new(1.1, 64, last_tokens.clone()))
        .push_sampler(SampleFreqPresence::new(0.05, 0.1, 64, last_tokens.clone()))
        .push_sampler(SampleTemperature::new(0.8))
        .push_sampler(SampleFlatBias::new(&[(3, f32::NEG_INFINITY)]))
        .push_sampler(SampleMirostat1::<u32, f32, StdRng>::new(
            4,
            5.0,
            0.1,
            60,
            10.0,
            Box::new(RngBox::new_seedable(Some(123))),
        ));
    last_tokens
        .write()
        .expect("Couldn't get write lock")
        .push(3);
    assert_eq!(sc.sample_token(&mut logits)?, Some(2));
    last_tokens
        .write()
        .expect("Couldn't get write lock")
        .push(2);
    assert_eq!(sc.sample_token(&mut logits2)?, Some(1));
    Ok(())
}
