use anyhow::Result;

use crate::prelude::*;

pub const T1: &[f32] = &[0.1, 0.2, 0.3, 0.4];
pub const TE1: &[f32] = &[0.4, 0.3, 0.2, 0.1];

pub type TestValidator<S> = fn(&mut S, &mut Logits, &[f32]);

fn test_sampler_ll<S: Sampler>(
    use_ln: bool,
    use_sm: bool,
    res: &mut dyn HasSamplerResources,
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    let mut logits = Logits::try_from_iter(input.iter().map(|i| if use_ln { i.ln() } else { *i }))
        .expect("Bad logits");
    if use_sm {
        logits.ensure_softmax().expect("Softmax failed");
    }
    let result_logits = sampler.sample(res, &mut logits).expect("Sampler error");
    vf(sampler, result_logits, expected)
}

fn test_sampler<S: Sampler>(
    res: &mut dyn HasSamplerResources,
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(true, true, res, sampler, input, expected, vf)
}

fn test_sampler_no_sm<S: Sampler>(
    res: &mut dyn HasSamplerResources,
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(true, false, res, sampler, input, expected, vf)
}

fn test_sampler_raw<S: Sampler>(
    res: &mut dyn HasSamplerResources,
    sampler: &mut S,
    input: &[f32],
    expected: &[f32],
    vf: TestValidator<S>,
) {
    test_sampler_ll(false, false, res, sampler, input, expected, vf)
}

fn validate(_sampler: &mut impl Sampler, logits: &mut Logits, expected: &[f32]) {
    let result = logits
        .iter()
        .zip(expected.iter())
        .map(|(l, e)| (l.prob - e).abs())
        .collect::<Vec<_>>();
    let lprobs = logits.iter().map(|i| i.prob).collect::<Vec<_>>();
    assert_eq!(
        result.len(),
        expected.len(),
        "result length mismatch: {lprobs:?} vs expected {expected:?}"
    );
    assert!(
        result.iter().all(|i| *i < 0.00001),
        "result {result:?} not within tolerance: {lprobs:?} vs expected {expected:?}"
    )
}

fn validate_sm(sampler: &mut impl Sampler, logits: &mut Logits, expected: &[f32]) {
    validate(
        sampler,
        logits.ensure_softmax().expect("Softmax failed"),
        expected,
    );
}

#[allow(dead_code)]
fn validate_sorted(sampler: &mut impl Sampler, logits: &mut Logits, expected: &[f32]) {
    validate(
        sampler,
        logits.ensure_sorted().expect("Sort failed"),
        expected,
    );
}

fn validate_eq(_sampler: &mut impl Sampler, logits: &mut Logits, expected: &[f32]) {
    assert_eq!(logits.iter().map(|l| l.logit).collect::<Vec<_>>(), expected)
}

fn do_test_greedy(it: impl Iterator<Item = f32>, expected: Option<u32>) -> Result<()> {
    assert_eq!(
        Logits::try_from_iter(it)?
            .sample_token(&mut NilSamplerResources, &mut SampleGreedy::new())?,
        expected
    );
    Ok(())
}

#[test]
fn test_chain1() -> anyhow::Result<()> {
    let mut res = NilSamplerResources;
    let mut logits = Logits::try_from_iter(T1.iter().copied())?;

    let mut sc = SamplerChain::new()
        + SampleFlatBias::new([(3, f32::NEG_INFINITY)])
        + SampleFlatBias::new([(2, f32::NEG_INFINITY)])
        + SampleGreedy::new();

    assert_eq!(sc.sample_token(&mut res, &mut logits)?, Some(1));
    Ok(())
}

#[test]
fn test_chain2() -> Result<()> {
    use rand::SeedableRng;
    let mut res = SimpleSamplerResources::new(
        Some(Box::new(rand::rngs::StdRng::seed_from_u64(123))),
        Some(vec![]),
    );
    let mut logits = Logits::try_from_iter(T1.iter().copied())?;
    let mut logits2 = logits.clone();

    let mut sc = SamplerChain::new()
        + SampleFlatBias::new([(3, f32::NEG_INFINITY)])
        + SampleRepetition::new(1.1, 64)
        + SampleFreqPresence::new(0.05, 0.1, 64)
        + SampleTemperature::new(0.8)
        + SampleMirostat1::new(4, 5.0, 0.1);
    res.with_last_tokens_mut(&mut |tokens| tokens.push(3u32))?;

    assert_eq!(sc.sample_token(&mut res, &mut logits)?, Some(2));

    res.with_last_tokens_mut(&mut |tokens| tokens.push(2u32))?;
    assert_eq!(sc.sample_token(&mut res, &mut logits2)?, Some(1));
    Ok(())
}

#[test]
fn test_resources() -> Result<()> {
    use rand::SeedableRng;
    let mut res = SimpleSamplerResources::new(
        Some(Box::new(rand::rngs::StdRng::seed_from_u64(123))),
        Some(vec![0u32]),
    );

    let mut derp = 0;
    res.with_rng_mut(&mut |rng| {
        derp = rng.next_u32();
    })?;
    res.with_rng_mut(&mut |rng| {
        derp = rng.next_u32();
    })?;
    Ok(())
}

mod sampler {
    use super::*;

    #[test]
    fn test_greedy() -> Result<()> {
        do_test_greedy(T1.iter().copied(), Some(3))?;
        do_test_greedy(T1.iter().rev().copied(), Some(0))
    }

    #[test]
    fn test_top_k() {
        let mut res = NilSamplerResources;
        test_sampler(
            &mut res,
            &mut SampleTopK::new(1, 0),
            T1,
            &TE1[0..1],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleTopK::new(3, 0),
            T1,
            &TE1[0..3],
            validate,
        );
    }

    #[test]
    fn test_top_p() {
        let mut res = NilSamplerResources;
        test_sampler(
            &mut res,
            &mut SampleTopP::new(0.0, 1),
            T1,
            &TE1[0..1],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleTopP::new(0.7, 1),
            T1,
            &TE1[0..2],
            validate,
        );
        test_sampler(&mut res, &mut SampleTopP::new(1.0, 1), T1, TE1, validate);
    }

    #[test]
    fn test_min_p() {
        const TINP: &[f32] = &[2.0, 1.0, 0.5, 0.25, 0.1];
        const TEXP: &[f32] = &[0.5194805, 0.25974026, 0.12987013, 0.064935066, 0.025974026];

        let mut res = NilSamplerResources;
        test_sampler(
            &mut res,
            &mut SampleMinP::new(2.0, 1),
            TINP,
            &TEXP[0..1],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleMinP::new(0.2, 1),
            TINP,
            &TEXP[0..3],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleMinP::new(0.0001, 1),
            TINP,
            TEXP,
            validate,
        );
    }

    #[test]
    fn test_top_a() {
        const TINP: &[f32] = &[2.0, 1.0, 0.5, 0.25, 0.1];
        const TEXP: &[f32] = &[0.5194805, 0.25974026, 0.12987013, 0.064935066, 0.025974026];

        let mut res = NilSamplerResources;
        test_sampler(
            &mut res,
            &mut SampleTopA::new(8.0, 2.0, 1),
            TINP,
            &TEXP[0..1],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleTopA::new(0.45, 2.0, 1),
            TINP,
            &TEXP[0..3],
            validate,
        );
        test_sampler(
            &mut res,
            &mut SampleTopA::new(0.0001, 2.0, 1),
            TINP,
            TEXP,
            validate,
        );
    }

    #[test]
    fn test_repetition() -> Result<()> {
        const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];
        const TER2: &[f32] = &[0.5, 0.5, 0.0, 0.0, 0.0];
        let mut res = SimpleSamplerResources::new(None, Some(vec![0]));

        test_sampler(
            &mut res,
            &mut SampleRepetition::new(50.0, 100),
            T,
            &[0.25, 0.25, 0.25, 0.25, 0.0],
            validate_sm,
        );
        res.with_last_tokens_mut(&mut |lt| {
            lt.push(1);
            lt.push(2);
        })?;
        test_sampler(
            &mut res,
            &mut SampleRepetition::new(50.0, 100),
            T,
            TER2,
            validate_sm,
        );
        res.with_last_tokens_mut(&mut |lt| {
            lt.push(0);
            lt.push(0);
        })?;
        test_sampler(
            &mut res,
            &mut SampleRepetition::new(50.0, 100),
            T,
            TER2,
            validate_sm,
        );
        Ok(())
    }

    #[test]
    fn test_freq_presence() -> Result<()> {
        const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];
        let mut res = SimpleSamplerResources::new(None, Some(vec![0]));

        test_sampler(
            &mut res,
            &mut SampleFreqPresence::new(5.0, 5.0, 100),
            T,
            &[0.249997, 0.249997, 0.249997, 0.249997, 0.000011],
            validate_sm,
        );
        res.with_last_tokens_mut(&mut |lt| {
            lt.push(1);
            lt.push(2);
        })?;
        test_sampler(
            &mut res,
            &mut SampleFreqPresence::new(5.0, 5.0, 100),
            T,
            &[0.499966, 0.499966, 0.000023, 0.000023, 0.000023],
            validate_sm,
        );
        res.with_last_tokens_mut(&mut |lt| {
            lt.push(0);
            lt.push(0);
        })?;
        test_sampler(
            &mut res,
            &mut SampleFreqPresence::new(5.0, 5.0, 100),
            T,
            &[0.499977, 0.499977, 0.000023, 0.000023, 0.0],
            validate_sm,
        );
        Ok(())
    }

    #[test]
    fn test_sequence_repetition() -> Result<()> {
        const T: &[f32] = &[0.2, 0.2, 0.2, 0.2, 0.2];
        let mut res = SimpleSamplerResources::new(None, Some(vec![0, 1, 2, 3, 0, 1, 2]));

        test_sampler(
            &mut res,
            &mut SampleSeqRepetition::default().min_length(3),
            T,
            &[0.2, 0.2, 0.2, 0.2, 0.2],
            validate_sm,
        );

        test_sampler(
            &mut res,
            &mut SampleSeqRepetition::default()
                .min_length(3)
                .flat_penalty(5.0),
            T,
            &[0.249579, 0.249579, 0.249579, 0.001681, 0.249579],
            validate_sm,
        );

        test_sampler(
            &mut res,
            &mut SampleSeqRepetition::default()
                .min_length(3)
                .stacking_penalty(1.25),
            T,
            &[0.249579, 0.249579, 0.249579, 0.001681, 0.249579],
            validate_sm,
        );

        let mut res = SimpleSamplerResources::new(None, Some(vec![0, 4, 2, 3, 0, 1, 2]));

        test_sampler(
            &mut res,
            &mut SampleSeqRepetition::default()
                .min_length(3)
                .tolerance(1)
                .stacking_penalty(1.25),
            T,
            &[0.249579, 0.249579, 0.249579, 0.001681, 0.249579],
            validate_sm,
        );

        Ok(())
    }

    #[test]
    fn test_locally_typical() {
        let mut res = NilSamplerResources;
        test_sampler_no_sm(
            &mut res,
            &mut SampleLocallyTypical::new(0.5, 1),
            &[0.97, 0.01, 0.01, 0.01],
            &[0.97],
            validate,
        );
        test_sampler_no_sm(
            &mut res,
            &mut SampleLocallyTypical::new(0.5, 1),
            &[0.4, 0.2, 0.2, 0.2],
            &[0.2, 0.2, 0.2],
            validate,
        );
    }

    #[test]
    fn test_tail_free() {
        const T: &[f32] = &[0.1, 0.15, 0.2, 0.25, 0.3];
        let mut res = NilSamplerResources;

        test_sampler_no_sm(
            &mut res,
            &mut SampleTailFree::new(0.25, 1),
            T,
            &[0.3],
            validate,
        );
        test_sampler_no_sm(
            &mut res,
            &mut SampleTailFree::new(0.75, 1),
            T,
            &[0.3, 0.25],
            validate,
        );
        test_sampler_no_sm(
            &mut res,
            &mut SampleTailFree::new(0.99, 1),
            T,
            &[0.3, 0.25],
            validate,
        );
    }

    #[test]
    fn test_flat_bias() {
        const T: &[f32] = &[0.1, 0.15, 0.2, 0.25, 0.3];
        let mut res = NilSamplerResources;

        test_sampler_raw(
            &mut res,
            &mut SampleFlatBias::new([(0, f32::NEG_INFINITY)]),
            T,
            &[f32::NEG_INFINITY, 0.15, 0.2, 0.25, 0.3],
            validate_eq,
        );
        test_sampler_raw(
            &mut res,
            &mut SampleFlatBias::new([(3, f32::NEG_INFINITY)]),
            T,
            &[0.1, 0.15, 0.2, f32::NEG_INFINITY, 0.3],
            validate_eq,
        );
    }

    #[test]
    fn test_rand_distrib() -> Result<()> {
        use rand::SeedableRng;
        let mut res = SimpleSamplerResources::new(
            Some(Box::new(rand::rngs::StdRng::seed_from_u64(123))),
            None,
        );
        let mut sampler = SampleRandDistrib::new();
        assert_eq!(
            Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(0)
        );
        assert_eq!(
            Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn test_mirostat1() -> Result<()> {
        use rand::SeedableRng;
        let mut res = SimpleSamplerResources::new(
            Some(Box::new(rand::rngs::StdRng::seed_from_u64(123))),
            None,
        );
        let mut sampler = SampleMirostat1::new(3, 5.0, 0.1);
        assert_eq!(
            Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(0)
        );
        let mut sampler = SampleMirostat1::new(3, 5.0, 0.1);
        assert_eq!(
            Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn test_mirostat2() -> Result<()> {
        use rand::SeedableRng;
        let mut res = SimpleSamplerResources::new(
            Some(Box::new(rand::rngs::StdRng::seed_from_u64(123))),
            None,
        );
        let mut sampler = SampleMirostat2::new(5.0, 0.1);
        assert_eq!(
            Logits::try_from_iter([1.0f32, 0.0, 0.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(0)
        );
        let mut sampler = SampleMirostat2::new(5.0, 0.1);
        assert_eq!(
            Logits::try_from_iter([0.0f32, 0.0, 1.0].into_iter().map(|i| i.ln()))?
                .sample_token(&mut res, &mut sampler)?,
            Some(2)
        );
        Ok(())
    }
}

mod configure {
    use super::*;

    use crate::configure::*;

    #[test]
    fn test_parse_uint() -> Result<()> {
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::UInt, "1")?,
            SamplerOptionValue::UInt(1)
        );
        assert!(SamplerOptionValue::parse_value(SamplerOptionType::UInt, "-1").is_err());
        assert!(SamplerOptionValue::parse_value(SamplerOptionType::UInt, "derp").is_err());
        Ok(())
    }

    #[test]
    fn test_parse_float() -> Result<()> {
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, "1")?,
            SamplerOptionValue::Float(1.0)
        );
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, "-1")?,
            SamplerOptionValue::Float(-1.0)
        );
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, ".1")?,
            SamplerOptionValue::Float(0.1)
        );
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, "-.1")?,
            SamplerOptionValue::Float(-0.1)
        );
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, "inf")?,
            SamplerOptionValue::Float(f64::INFINITY)
        );
        assert_eq!(
            SamplerOptionValue::parse_value(SamplerOptionType::Float, "-inf")?,
            SamplerOptionValue::Float(f64::NEG_INFINITY)
        );
        assert!(SamplerOptionValue::parse_value(SamplerOptionType::UInt, "derp").is_err());
        Ok(())
    }

    #[test]
    fn test_set_get_options() -> Result<()> {
        let mut samp = SampleTemperature::new(5.0);
        assert_eq!(
            ConfigurableSampler::<u32, f32>::get_option(&samp, "temperature")?,
            SamplerOptionValue::Float(5.0)
        );
        assert!(ConfigurableSampler::<u32, f32>::set_option(
            &mut samp,
            "temperature",
            SamplerOptionValue::Bool(true)
        )
        .is_err());
        ConfigurableSampler::<u32, f32>::set_option(
            &mut samp,
            "temperature",
            SamplerOptionValue::Float(8.0),
        )?;
        assert_eq!(
            ConfigurableSampler::<u32, f32>::get_option(&samp, "temperature")?,
            SamplerOptionValue::Float(8.0)
        );
        Ok(())
    }

    #[test]
    fn test_config_from_str1() -> Result<()> {
        let mut samp = SampleTemperature::new(5.0);

        ConfigurableSampler::<u32, f32>::configure(&mut samp, "7.0")?;
        assert_eq!(
            ConfigurableSampler::<u32, f32>::get_option(&samp, "temperature")?,
            SamplerOptionValue::Float(7.0)
        );
        assert!(ConfigurableSampler::<u32, f32>::configure(&mut samp, "xyz=2.0").is_err());
        ConfigurableSampler::<u32, f32>::configure(&mut samp, " temperature =   7.0 ")?;
        assert_eq!(
            ConfigurableSampler::<u32, f32>::get_option(&samp, "temperature")?,
            SamplerOptionValue::Float(7.0)
        );
        Ok(())
    }

    #[test]
    fn test_config_from_str2() -> Result<()> {
        let mut samp = SampleFreqPresence::default();

        samp.configure("frequency_penalty=inf : presence_penalty=-inf : last_n =69")?;
        assert_eq!(
            samp.get_option("frequency_penalty")?,
            SamplerOptionValue::Float(f64::INFINITY)
        );
        assert_eq!(
            samp.get_option("presence_penalty")?,
            SamplerOptionValue::Float(f64::NEG_INFINITY)
        );
        assert_eq!(samp.get_option("last_n")?, SamplerOptionValue::UInt(69));
        samp.configure("f=-inf : pres=inf : last =96")?;
        assert_eq!(
            samp.get_option("frequency_penalty")?,
            SamplerOptionValue::Float(f64::NEG_INFINITY)
        );
        assert_eq!(
            samp.get_option("presence_penalty")?,
            SamplerOptionValue::Float(f64::INFINITY)
        );
        assert_eq!(samp.get_option("last_n")?, SamplerOptionValue::UInt(96));
        Ok(())
    }
}

mod build {
    use super::*;

    use crate::configure::*;

    #[test]
    fn test_build1() -> Result<()> {
        let mut ss: SamplerChainBuilder<usize, f32> = SamplerChainBuilder::from([
            (
                "rep".to_string(),
                SamplerSlot::new_chain(|| Box::new(SampleRepetition::new(0.0, 0)), []),
            ),
            (
                "freqpres".to_string(),
                SamplerSlot::new_single(
                    || Box::new(SampleFreqPresence::new(0.0, 0.0, 0)),
                    Option::<SampleFreqPresence>::None,
                ),
            ),
            (
                "greedy".to_string(),
                SamplerSlot::new_static(|| Box::new(SampleGreedy::new())),
            ),
        ]);

        ss.configure("rep", "penalty=1.1:last_n=64")?;
        ss.configure("rep", "penalty=1.1:last_n=64")?;
        ss.configure("freqpres", "frequency=.5")?;
        ss.configure("freqpres", "last_n=4")?;

        let mut sc = ss.into_chain();

        let mut res = SimpleSamplerResources::new(None, Some(vec![0, 1, 2, 3, 3, 0, 0]));
        let mut logits = Logits::try_from_iter([0.2, 0.2, 0.2, 0.2].into_iter())?;
        let tok = sc.sample_token(&mut res, &mut logits)?;
        assert_eq!(tok, Some(1));

        Ok(())
    }
}
