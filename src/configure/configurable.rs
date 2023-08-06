use std::borrow::Cow;

use anyhow::Result;
use num_traits::NumCast;

use super::*;

/// Configurable samplers implement this trait. "Configurable" means
/// they allow access to their their options by key/type and allow configuration
/// based on descriptions.
///
/// There are default implementations for for all the methods, so in the general
/// case you will only need to implement the `OPTIONS` constant with the option
/// definitions.
pub trait ConfigurableSampler<UI = u32, F = f32>: HasSamplerMetadata<UI, F>
where
    UI: ConfigurableNumValue,
    F: ConfigurableNumValue,
{
    /// Given an option key and [SamplerOptionValue] attempts to set the option
    /// to the specified value.
    ///
    /// The default implementation will call [Self::pre_set_option] before setting
    /// the option and [Self::post_set_option] afterward. For an example of
    /// the latter you can look at the Mirostat samplers: when `tau` is set, they'll
    /// automatically set `mu` to `tau * 2`.
    fn set_option(&mut self, key: &str, val: SamplerOptionValue) -> Result<()> {
        configurable_sampler::set_option(self, key, val)?;
        Ok(())
    }

    /// Called before an option is set and is passed a mutable reference
    /// to the [SamplerOptionValue]. It is also passed an index into
    /// the options definition list.
    ///
    /// The default implementation is a no-op.
    #[allow(unused_variables)]
    fn pre_set_option(
        &mut self,
        md: &SamplerOptionMetadata,
        val: &mut SamplerOptionValue,
    ) -> Result<()> {
        Ok(())
    }

    /// Called before an option is set is passed an index into
    /// the options definition list.
    ///
    /// The default implementation is a no-op.
    #[allow(unused_variables)]
    fn post_set_option(&mut self, md: &SamplerOptionMetadata) -> Result<()> {
        Ok(())
    }

    /// Gets an option by name.
    fn get_option(&self, key: &str) -> Result<SamplerOptionValue> {
        configurable_sampler::get_option(self, key)
    }

    /// Updates a sampler's configurable options based on a string in the
    /// format:
    ///
    /// `key1=value1:key2=value2:keyN=valueN`
    ///
    /// The key be a prefix of the option name as long as it's not
    /// ambiguous. It's also possible to just specify the value,
    /// which is equivalent to `=value` (i.e. a blank key name).
    ///
    /// Values in this default implementation cannot contain `=` or `:`
    /// and whitespace at the beginning and end of parts are stripped.
    fn configure(&mut self, s: &str) -> Result<()> {
        configurable_sampler::configure(self, s)?;
        Ok(())
    }
}

/// Since Rust traits don't allow calling base default methods from
/// a more specific implementation, the [ConfigurableSampler] trait
/// default methods are implemented in terms of the functions
/// in this submodule.
///
/// See the trait for descriptions of the methods.
pub mod configurable_sampler {
    use super::*;

    pub fn set_option<'a, CS, UI, F>(
        slf: &'a mut CS,
        key: &str,
        mut val: SamplerOptionValue,
    ) -> Result<&'a mut CS>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let key = key.trim();
        let (omd, optidx) = {
            let opts = slf.sampler_options_mut();
            if let (omd, Some(optidx)) = opts.find_option_definition(key)? {
                (omd, optidx)
            } else {
                Err(ConfigureSamplerError::CannotAccessOptionValue(
                    key.to_string(),
                ))?
            }
        };

        slf.pre_set_option(&omd, &mut val)?;

        let mut opts = slf.sampler_options_mut();
        let acc = opts[optidx]
            .1
            .take()
            .ok_or_else(|| ConfigureSamplerError::CannotAccessOptionValue(key.to_string()))?;

        match (acc, val) {
            (SamplerOptionValueMut::Float(rv), SamplerOptionValue::Float(v)) => {
                *rv = F::from_f64(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?
            }

            (SamplerOptionValueMut::UInt(rv), SamplerOptionValue::UInt(v)) => {
                *rv = UI::from_u64(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?
            }
            (SamplerOptionValueMut::Bool(rv), SamplerOptionValue::Bool(v)) => *rv = v,
            (SamplerOptionValueMut::String(rv), SamplerOptionValue::String(v)) => {
                *rv = Cow::from(v.to_string())
            }
            _ => Err(ConfigureSamplerError::UnknownOrBadType(key.to_string()))?,
        }
        slf.post_set_option(&omd)?;
        Ok(slf)
    }

    pub fn get_option<'a, CS, UI, F>(slf: &'a CS, key: &str) -> Result<SamplerOptionValue<'a>>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let key = key.trim();

        let mut opts = slf.sampler_options();

        let (_omd, Some(optidx)) = opts.find_option_definition(key)? else {
            Err(ConfigureSamplerError::CannotAccessOptionValue(key.to_string()))?
        };

        Ok(match opts[optidx].1.take().expect("Impossible") {
            SamplerOptionValue::UInt(v) => SamplerOptionValue::UInt(
                <u64 as NumCast>::from(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?,
            ),
            SamplerOptionValue::Float(v) => SamplerOptionValue::Float(
                <f64 as NumCast>::from(v)
                    .ok_or_else(|| ConfigureSamplerError::ConversionFailure(key.to_string()))?,
            ),
            SamplerOptionValue::Bool(v) => SamplerOptionValue::Bool(v),
            SamplerOptionValue::String(v) => SamplerOptionValue::String(Cow::from(v.to_string())),
        })
    }

    pub fn configure<CS, UI, F>(slf: &mut CS, s: &str) -> Result<()>
    where
        CS: ConfigurableSampler<UI, F> + HasSamplerMetadata<UI, F> + ?Sized,
        UI: ConfigurableNumValue,
        F: ConfigurableNumValue,
    {
        let opts = SamplerOptions::from(
            slf.sampler_options_mut()
                .iter()
                .map(|(md, acc)| (md.clone(), acc.is_some().then_some(()))),
        );
        s.trim()
            .split(':')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .try_for_each(|kv| {
                let (k, v) = kv.split_once('=').unwrap_or(("", kv));
                let (omd, Some(_)) = opts.find_option_definition(k)? else {
                    Err(ConfigureSamplerError::UnknownOrBadType(k.to_string()))?
                };

                slf.set_option(
                    omd.key,
                    SamplerOptionValue::parse_value(omd.option_type, v.trim())?,
                )?;
                anyhow::Ok(())
            })?;
        Ok(())
    }
}
