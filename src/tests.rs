use super::*;

const T1: &[f32] = &[0.1, 0.2, 0.3, 0.4];
const TE1: &[f32] = &[0.4, 0.3, 0.2, 0.1];

type TestValidator<S> = fn(&mut S, &mut Logits<u32, f32>, &[f32]);

fn test_sampler<S: Sampler<u32, f32>>(
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    let mut logits = Logits::from(input.iter().map(|i| i.ln()));
    logits.softmax();
    let result_logits = sampler.sample(&mut logits);
    vf(sampler, result_logits, expected);
}

fn test_sampler_no_sm<S: Sampler<u32, f32>>(
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    let mut logits = Logits::from(input.iter().map(|i| i.ln()));
    let result_logits = sampler.sample(&mut logits);
    vf(sampler, result_logits, expected);
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
    validate(sampler, logits.softmax(), expected);
}

fn do_test_greedy(it: impl Iterator<Item = f32>, expected: Option<u32>) {
    let mut logits = Logits::from(it);
    let mut g = SampleGreedy::new();
    g.sample(&mut logits);
    assert_eq!(g.get_token_id(), expected);
}

#[test]
fn test_greedy() {
    do_test_greedy(T1.iter().copied(), Some(3));
    do_test_greedy(T1.iter().rev().copied(), Some(0));
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
        &mut SampleRepetition::new(50.0, &[0]),
        T,
        &[0.25, 0.25, 0.25, 0.25, 0.0],
        validate_sm,
    );
    test_sampler(
        &mut SampleRepetition::new(50.0, &[0, 1, 2]),
        T,
        TER2,
        validate_sm,
    );
    test_sampler(
        &mut SampleRepetition::new(50.0, &[0, 1, 2, 0, 0]),
        T,
        TER2,
        validate_sm,
    );
}

#[test]
fn test_freq_presence() {
    const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];

    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, &[0]),
        T,
        &[0.249997, 0.249997, 0.249997, 0.249997, 0.000011],
        validate_sm,
    );
    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, &[0, 1, 2]),
        T,
        &[0.499966, 0.499966, 0.000023, 0.000023, 0.000023],
        validate_sm,
    );
    test_sampler(
        &mut SampleFreqPresence::new(5.0, 5.0, &[0, 1, 2, 0, 0]),
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
