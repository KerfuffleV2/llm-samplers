use std::ops::{Deref, DerefMut};

use num_traits::{Float, FromPrimitive, PrimInt, ToPrimitive};

pub trait CanTokenId: PrimInt + FromPrimitive + ToPrimitive {}

impl<T: PrimInt + FromPrimitive + ToPrimitive> CanTokenId for T {}

#[derive(Debug, Clone, PartialEq)]
pub struct Logit<TID, L> {
    pub token_id: TID,
    pub logit: L,
    pub prob: L,
}

#[derive(Debug, Clone)]
pub struct Logits<TID, L> {
    sorted: bool,
    logits: Vec<Logit<TID, L>>,
}

impl<TID, L> Deref for Logits<TID, L> {
    type Target = Vec<Logit<TID, L>>;

    fn deref(&self) -> &Self::Target {
        &self.logits
    }
}

impl<TID, L> DerefMut for Logits<TID, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.logits
    }
}

impl<L: Float, I: IntoIterator<Item = L>> From<I> for Logits<u32, L> {
    fn from(value: I) -> Self {
        Self {
            sorted: false,
            logits: Vec::from_iter(value.into_iter().enumerate().map(|(tid, logit)| Logit {
                token_id: tid as u32,
                logit,
                prob: L::zero(),
            })),
        }
    }
}

impl<TID: PrimInt, L: Float> Logits<TID, L> {
    pub fn get_sorted(&self) -> bool {
        self.sorted
    }

    pub fn set_sorted(&mut self, is_sorted: bool) -> &mut Self {
        self.sorted = is_sorted;
        self
    }

    pub fn ensure_sorted(&mut self) -> &mut Self {
        if self.get_sorted() {
            return self;
        }
        self.logits.as_mut_slice().sort_by(|a, b| {
            a.logit
                .partial_cmp(&b.logit)
                .expect("Comparison failed!")
                .reverse()
        });
        self.set_sorted(true);
        self
    }

    pub fn softmax(&mut self) -> &mut Self {
        if self.is_empty() {
            return self;
        }
        self.ensure_sorted();
        let max_l = self[0].logit;
        let cum_sum = self.iter_mut().fold(L::zero(), |cs, l| {
            let p = (l.logit - max_l).exp();
            l.prob = p;
            cs + p
        });
        self.iter_mut().for_each(|l| l.prob = l.prob / cum_sum);
        self
    }

    pub fn sample<S: Sampler<TID, L>>(&mut self, sampler: &mut S) -> &mut Self {
        sampler.sample(self)
    }

    pub fn sample_token<S: SampleToken<TID, L>>(&mut self, sampler: &mut S) -> Option<TID> {
        sampler.sample_token(self)
    }
}

pub trait Sampler<TID: PrimInt, L: Float> {
    fn sample<'a>(&mut self, logits: &'a mut Logits<TID, L>) -> &'a mut Logits<TID, L>;
}

pub trait SampleToken<TID: PrimInt, L: Float> {
    fn sample_token(&mut self, logits: &mut Logits<TID, L>) -> Option<TID>;
}
