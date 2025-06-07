use super::Vec;
use crate::{Collect, Mutation};

pub(super) trait SpecToVec<'gc> {
    fn to_vec(mc: &Mutation<'gc>, s: &[Self]) -> Vec<'gc, Self>
    where
        Self: Sized;

    fn clone_into_vec(mc: &Mutation<'gc>, s: &[Self], target: &mut Vec<'gc, Self>)
    where
        Self: Sized;
}

impl<'gc, T: Clone + Collect<'gc> + 'gc> SpecToVec<'gc> for T {
    #[inline]
    default fn to_vec(mc: &Mutation<'gc>, s: &[Self]) -> Vec<'gc, Self> {
        struct DropGuard<'a, 'gc, T> {
            vec: &'a mut Vec<'gc, T>,
            num_init: usize,
        }
        impl<'a, 'gc, T> Drop for DropGuard<'a, 'gc, T> {
            #[inline]
            fn drop(&mut self) {
                // SAFETY:
                // items were marked initialized in the loop below
                unsafe {
                    self.vec.set_len(self.num_init);
                }
            }
        }

        let mut vec = Vec::with_capacity(mc, s.len());
        let mut guard = DropGuard { vec: &mut vec, num_init: 0 };
        let slots = guard.vec.spare_capacity_mut();
        // .take(slots.len()) is necessary for LLVM to remove bounds checks
        // and has better codegen than zip.
        for (i, b) in s.iter().enumerate().take(slots.len()) {
            guard.num_init = i;
            slots[i].write(b.clone());
        }
        core::mem::forget(guard);
        // SAFETY:
        // the vec was allocated and initialized above to at least this length.
        unsafe { vec.set_len(s.len()) };
        vec
    }

    default fn clone_into_vec(mc: &Mutation<'gc>, s: &[Self], target: &mut Vec<'gc, Self>) {
        // drop anything in target that will not be overwritten
        target.truncate(s.len());

        // target.len <= self.len due to the truncate above, so the
        // slices here are always in-bounds.
        let (init, tail) = s.split_at(target.len());

        // reuse the contained values' allocations/resources.
        target.clone_from_slice(init);
        target.extend_from_slice(mc, tail);
    }
}

impl<'gc, T: Copy + Collect<'gc> + 'gc> SpecToVec<'gc> for T {
    #[inline]
    fn to_vec(mc: &Mutation<'gc>, s: &[Self]) -> Vec<'gc, Self> {
        let mut v = Vec::with_capacity(mc, s.len());
        // SAFETY:
        // allocated above with the capacity of `s`, and initialize to `s.len()` in
        // ptr::copy_to_non_overlapping below.
        unsafe {
            s.as_ptr().copy_to_nonoverlapping(v.as_mut_ptr(), s.len());
            v.set_len(s.len());
        }
        v
    }

    fn clone_into_vec(mc: &Mutation<'gc>, s: &[Self], target: &mut Vec<'gc, Self>)
    where
        Self: Sized,
    {
        target.clear();
        target.extend_from_slice(mc, s);
    }
}

pub trait Sealed {}
pub trait ConvertVec<'gc>: Sealed {
    type Item: Collect<'gc> + Clone;

    fn to_vec(&self, mc: &Mutation<'gc>) -> Vec<'gc, Self::Item>;

    fn clone_into_vec(&self, mc: &Mutation<'gc>, target: &mut Vec<'gc, Self::Item>);
}

impl<T> Sealed for [T] {}
impl<'gc, T: Collect<'gc> + Clone + 'gc> ConvertVec<'gc> for [T] {
    type Item = T;

    #[inline]
    fn to_vec(&self, mc: &Mutation<'gc>) -> Vec<'gc, Self::Item> {
        SpecToVec::to_vec(mc, self)
    }

    fn clone_into_vec(&self, mc: &Mutation<'gc>, target: &mut Vec<'gc, Self::Item>) {
        SpecToVec::clone_into_vec(mc, self, target)
    }
}
