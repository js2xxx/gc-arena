use core::{cmp, iter::TrustedLen, ptr};

use crate::{Collect, Mutation};

use super::{SpecExtend, Vec, min_non_zero_cap};

pub trait SpecFromIter<'gc, T: Collect<'gc> + 'gc, I> {
    fn from_iter(mc: &Mutation<'gc>, iter: I) -> Self;
}

impl<'gc, T, I> SpecFromIter<'gc, T, I> for Vec<'gc, T>
where
    I: Iterator<Item = T>,
    T: Collect<'gc> + 'gc,
{
    default fn from_iter(mc: &Mutation<'gc>, iter: I) -> Self {
        SpecFromIterNested::from_iter(mc, iter)
    }
}

// TODO: Specialize on vec::IntoIter.
//
// impl<'gc, T> SpecFromIter<'gc, T, super::IntoIter<'gc, T>> for Vec<'gc, T>
// where
//     T: Collect<'gc> + 'gc,
// {
//     #[track_caller]
//     fn from_iter(mc: &Mutation<'gc>, iter: super::IntoIter<'gc, T>) -> Self {
//         iter.into_vec(mc)
//     }
// }

pub(super) trait SpecFromIterNested<'gc, T, I> {
    fn from_iter(mc: &Mutation<'gc>, iter: I) -> Self;
}

impl<'gc, T, I> SpecFromIterNested<'gc, T, I> for Vec<'gc, T>
where
    I: Iterator<Item = T>,
    T: Collect<'gc> + 'gc,
{
    #[track_caller]
    default fn from_iter(mc: &Mutation<'gc>, mut iter: I) -> Self {
        // Unroll the first iteration, as the vector is going to be
        // expanded on this iteration in every case when the iterable is not
        // empty, but the loop in extend_desugared() is not going to see the
        // vector being full in the few subsequent loop iterations.
        // So we get better branch prediction.
        let mut vec = match iter.next() {
            None => return Vec::new(mc),
            Some(element) => {
                let (lower, _) = iter.size_hint();
                let initial_capacity =
                    cmp::max(min_non_zero_cap(size_of::<T>()), lower.saturating_add(1));
                let mut vector = Vec::with_capacity(mc, initial_capacity);
                unsafe {
                    // SAFETY: We requested capacity at least 1
                    ptr::write(vector.as_mut_ptr(), element);
                    vector.set_len(1);
                }
                vector
            }
        };
        // must delegate to spec_extend() since extend() itself delegates
        // to spec_from for empty Vecs
        <Vec<T> as SpecExtend<'gc, T, I>>::extend(&mut vec, mc, iter);
        vec
    }
}

impl<'gc, T, I> SpecFromIterNested<'gc, T, I> for Vec<'gc, T>
where
    I: TrustedLen<Item = T>,
    T: Collect<'gc> + 'gc,
{
    #[track_caller]
    fn from_iter(mc: &Mutation<'gc>, iter: I) -> Self {
        let mut vec = match iter.size_hint() {
            (_, Some(upper)) => Vec::with_capacity(mc, upper),
            // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
            // are more than `usize::MAX` elements.
            // Since the previous branch would eagerly panic if the capacity is too large
            // (via `with_capacity`) we do the same here.
            _ => panic!("capacity overflow"),
        };
        // reuse extend specialization for TrustedLen
        vec.extend(mc, iter);
        vec
    }
}
