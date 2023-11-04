use std::{
    any::{Any, TypeId},
    fmt::Debug,
    marker::PhantomData,
};

use crate::types::{CanLogit, CanTokenId, Logits, SamplerError};

pub struct RngResource<'a> {
    pub rng: &'a mut dyn rand::RngCore,
}

pub struct LastTokensResource<'a, T> {
    pub last_tokens: &'a [T],
}

pub enum ResourceValue<'a> {
    Any(&'a dyn Any),
    AnyMut(&'a mut dyn Any),
    AnySlice(AnySlice<'a>),
    AnySliceMut(AnySliceMut<'a>),
}

#[derive(Clone, Debug)]
pub struct AnySlice<'a> {
    tid: TypeId,
    len: usize,
    ptr: *const (),
    marker: PhantomData<&'a ()>,
}

impl<'a> AnySlice<'a> {
    pub fn from_slice<T: Any>(s: &'a [T]) -> Self {
        Self {
            len: s.len(),
            ptr: s.as_ptr() as *const (),
            tid: TypeId::of::<T>(),
            marker: PhantomData,
        }
    }

    pub fn try_as_slice<T: Any>(&self) -> Option<&'a [T]> {
        if TypeId::of::<T>() != self.tid {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) })
    }

    pub fn is<T: Any>(&self) -> bool {
        TypeId::of::<T>() == self.tid
    }
}

#[derive(Debug)]
pub struct AnySliceMut<'a> {
    tid: TypeId,
    len: usize,
    ptr: *mut (),
    marker: PhantomData<&'a mut ()>,
}

impl<'a> AnySliceMut<'a> {
    pub fn from_slice_mut<T: Any>(s: &'a mut [T]) -> Self {
        Self {
            len: s.len(),
            ptr: s.as_mut_ptr() as *mut (),
            tid: TypeId::of::<T>(),
            marker: PhantomData,
        }
    }

    pub fn try_into_mut_slice<T: Any>(self) -> Result<&'a mut [T], Self> {
        if TypeId::of::<T>() != self.tid {
            return Err(self);
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) })
    }

    pub fn try_as_mut_slice<T: Any>(&mut self) -> Option<&mut [T]> {
        if TypeId::of::<T>() != self.tid {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) })
    }

    pub fn is<T: Any>(&self) -> bool {
        TypeId::of::<T>() == self.tid
    }
}

pub trait IsResource<T>: Sized {
    fn get_resource(self) -> Option<T>;
}

impl<'a, 'b, T: Any> IsResource<&'b T> for &'b ResourceValue<'a> {
    fn get_resource(self) -> Option<&'b T> {
        match self {
            ResourceValue::Any(r) => r.downcast_ref(),
            _ => None,
        }
    }
}

impl<'a, 'b, T: Any> IsResource<&'b mut T> for &'b mut ResourceValue<'a> {
    fn get_resource(self) -> Option<&'b mut T> {
        match self {
            ResourceValue::AnyMut(r) => r.downcast_mut(),
            _ => None,
        }
    }
}

impl<'a, 'b, T: Any> IsResource<&'b [T]> for &'b ResourceValue<'a> {
    fn get_resource(self) -> Option<&'b [T]> {
        match self {
            ResourceValue::AnySlice(r) => r.try_as_slice(),
            _ => None,
        }
    }
}

impl<'a, 'b, T: Any> IsResource<&'b mut [T]> for &'b mut ResourceValue<'a> {
    fn get_resource(self) -> Option<&'b mut [T]> {
        match self {
            ResourceValue::AnySliceMut(r) => r.try_as_mut_slice(),
            _ => None,
        }
    }
}

impl<'a, T: Any> IsResource<&'a T> for ResourceValue<'a> {
    fn get_resource(self) -> Option<&'a T> {
        match self {
            ResourceValue::Any(r) => r.downcast_ref(),
            _ => None,
        }
    }
}

impl<'a, T: Any> IsResource<&'a mut T> for ResourceValue<'a> {
    fn get_resource(self) -> Option<&'a mut T> {
        match self {
            ResourceValue::AnyMut(r) => r.downcast_mut(),
            _ => None,
        }
    }
}

impl<'a, T: Any> IsResource<&'a [T]> for ResourceValue<'a> {
    fn get_resource(self) -> Option<&'a [T]> {
        match self {
            ResourceValue::AnySlice(r) => r.try_as_slice(),
            _ => None,
        }
    }
}

impl<'a, T: Any> IsResource<&'a mut [T]> for ResourceValue<'a> {
    fn get_resource(self) -> Option<&'a mut [T]> {
        match self {
            ResourceValue::AnySliceMut(r) => r.try_into_mut_slice().ok(),
            _ => None,
        }
    }
}

// pub trait SamplerResource {
//     type Resource;
//     type ResourceMut;

//     fn resource_name(&self) -> &'static str {
//         "unknown "
//     }

//     fn get_resource(&self) -> Result<&Self::Resource, SamplerError> {
//         Err(SamplerError::MissingResource(
//             self.resource_name().to_string(),
//         ))
//     }

//     fn get_resource_mut(&self) -> Result<&mut Self::ResourceMut, SamplerError> {
//         Err(SamplerError::MissingResource(
//             self.resource_name().to_string(),
//         ))
//     }
// }

/// Trait for providing resources to samplers.
pub trait HasSamplerResources: Debug {
    /// The token ID type for the sampler that will use these resources.
    type TokenId: Send + Sync + Clone;

    /// Allows a sampler to mutably access the RNG (if present).
    fn with_rng_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("rng".to_string()))
    }

    /// Allows a sampler to immutably access the last tokens (if present).
    fn with_last_tokens(&self, _fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("last_tokens".to_string()))
    }

    /// Allows a sampler to mutably access the last tokens (if present).
    fn with_last_tokens_mut(
        &mut self,
        _fun: &mut dyn FnMut(&mut Vec<Self::TokenId>),
    ) -> Result<(), SamplerError> {
        Err(SamplerError::MissingResource("last_tokens".to_string()))
    }

    fn get_resource(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Err(SamplerError::MissingResource(format!("dyn({key:?})")))
    }

    fn get_resource_mut(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Err(SamplerError::MissingResource(format!("dyn_mut({key:?})")))
    }

    fn with_resource(
        &self,
        key: &str,
        _fun: &mut dyn FnMut(&ResourceValue) -> Option<Box<dyn Any>>,
    ) -> Result<Option<Box<dyn Any>>, SamplerError> {
        Err(SamplerError::MissingResource(format!("dyn({key:?})")))
    }

    fn with_resource_mut(
        &mut self,
        key: &str,
        _fun: &mut dyn FnMut(&mut ResourceValue) -> Option<Box<dyn Any>>,
    ) -> Result<Option<Box<dyn Any>>, SamplerError> {
        Err(SamplerError::MissingResource(format!("dyn_mut({key:?})")))
    }
}

#[derive(Debug, Clone)]
/// Empty resource structure for use with samplers that don't require
/// any resources.
pub struct NilSamplerResources<TID = u32>(PhantomData<TID>);

impl<TID> Default for NilSamplerResources<TID> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<TID> NilSamplerResources<TID> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<TID: Debug + Send + Sync + Clone> HasSamplerResources for NilSamplerResources<TID> {
    type TokenId = TID;
}

impl HasSamplerResources for () {
    type TokenId = u32;
}

/// Simple resources that can provide an RNG and/or last tokens to samplers.
pub struct SimpleSamplerResources<TID = u32, L = f32> {
    pub(crate) rng: Option<Box<dyn rand::RngCore + Send + Sync>>,

    pub(crate) last_tokens: Option<Vec<TID>>,

    pub(crate) logits: Option<Logits<TID, L>>,
}

impl<TID: Debug, L: Debug> Debug for SimpleSamplerResources<TID, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources")
            .field("rng", &self.rng.is_some())
            .field("last_tokens", &self.last_tokens)
            .finish()
    }
}

impl<TID: CanTokenId, L: CanLogit> SimpleSamplerResources<TID, L> {
    pub fn new(
        rng: Option<Box<dyn rand::RngCore + Send + Sync>>,
        last_tokens: Option<Vec<TID>>,
    ) -> Self {
        Self {
            rng,
            last_tokens,
            logits: None,
        }
    }

    pub fn set_logits(&mut self, logits: Option<Logits<TID, L>>) {
        self.logits = logits;
    }
}

impl<TID: CanTokenId + 'static, L: CanLogit + 'static> HasSamplerResources
    for SimpleSamplerResources<TID, L>
{
    type TokenId = TID;

    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        self.rng.as_mut().map_or_else(
            || Err(SamplerError::MissingResource("rng".to_string())),
            |rng| {
                fun(rng);
                Ok(())
            },
        )
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        self.last_tokens.as_ref().map_or_else(
            || Err(SamplerError::MissingResource("last_tokens".to_string())),
            |lt| {
                fun(lt);
                Ok(())
            },
        )
    }

    fn with_last_tokens_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut Vec<Self::TokenId>),
    ) -> Result<(), SamplerError> {
        self.last_tokens.as_mut().map_or_else(
            || Err(SamplerError::MissingResource("last_tokens".to_string())),
            |lt| {
                fun(lt);
                Ok(())
            },
        )
    }

    fn get_resource(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Ok(match key {
            "logits" if self.logits.is_some() => ResourceValue::Any(self.logits.as_ref().unwrap()),
            "last_tokens" if self.last_tokens.is_some() => {
                ResourceValue::AnySlice(AnySlice::from_slice(self.last_tokens.as_ref().unwrap()))
            }
            _ => return Err(SamplerError::MissingResource(format!("dyn({key:?})"))),
        })
    }

    fn get_resource_mut(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Ok(match key {
            "logits" if self.logits.is_some() => {
                ResourceValue::AnyMut(self.logits.as_mut().unwrap())
            }
            _ => return Err(SamplerError::MissingResource(format!("dyn_mut({key:?})"))),
        })
    }
}

///////
///
///
use std::collections::HashMap;

/// Simple resources that can provide an RNG and/or last tokens to samplers.
pub struct SimpleSamplerResources2<'a, TID = u32, L = f32> {
    blah: std::marker::PhantomData<(TID, L)>,
    vals: HashMap<&'static str, ResourceValue<'a>>,
}

impl<'a, TID: Debug, L: Debug> Debug for SimpleSamplerResources2<'a, TID, L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SamplerResources").finish()
    }
}

impl<'a, TID: CanTokenId, L: CanLogit> SimpleSamplerResources2<'a, TID, L> {
    pub fn new<R: rand::RngCore + Send + Sync>(
        rng: Option<&'a mut R>,
        last_tokens: Option<&'a [TID]>,
        logits: Option<&'a mut Logits<TID, L>>,
    ) -> Self {
        let mut hm = HashMap::new();
        if let Some(mut v) = rng {
            hm.insert("rng", ResourceValue::AnyMut(v as &mut dyn Any));
        }
        if let Some(mut v) = logits {
            hm.insert("logits", ResourceValue::AnyMut(v as &mut dyn Any));
        }
        if let Some(mut v) = last_tokens {
            hm.insert(
                "last_tokens",
                ResourceValue::AnySlice(AnySlice::from_slice(v)),
            );
        }
        Self {
            blah: PhantomData,
            vals: hm,
        }
    }
}

impl<'a, TID: CanTokenId + 'static, L: CanLogit + 'static> HasSamplerResources
    for SimpleSamplerResources2<'a, TID, L>
{
    type TokenId = TID;

    fn with_rng_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut dyn rand::RngCore),
    ) -> Result<(), SamplerError> {
        todo!()
    }

    fn with_last_tokens(&self, fun: &mut dyn FnMut(&[Self::TokenId])) -> Result<(), SamplerError> {
        todo!()
    }

    fn with_last_tokens_mut(
        &mut self,
        fun: &mut dyn FnMut(&mut Vec<Self::TokenId>),
    ) -> Result<(), SamplerError> {
        todo!()
    }

    fn get_resource(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Ok(match key {
            "logits" if self.logits.is_some() => ResourceValue::Any(self.logits.as_ref().unwrap()),
            "last_tokens" if self.last_tokens.is_some() => {
                ResourceValue::AnySlice(AnySlice::from_slice(self.last_tokens.as_ref().unwrap()))
            }
            _ => return Err(SamplerError::MissingResource(format!("dyn({key:?})"))),
        })
    }

    fn get_resource_mut(&self, key: &str) -> Result<ResourceValue, SamplerError> {
        Ok(match key {
            "logits" if self.logits.is_some() => {
                ResourceValue::AnyMut(self.logits.as_mut().unwrap())
            }
            _ => return Err(SamplerError::MissingResource(format!("dyn_mut({key:?})"))),
        })
    }
}
