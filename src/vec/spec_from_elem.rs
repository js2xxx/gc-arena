use core::ptr;

use super::Vec;
use crate::{Collect, Gc, Mutation};

pub(super) trait SpecFromElem<'gc>: Sized {
    fn from_elem(elem: Self, n: usize, mc: &Mutation<'gc>) -> Vec<'gc, Self>;
}

impl<'gc, T: Clone + Collect<'gc>> SpecFromElem<'gc> for T {
    #[track_caller]
    default fn from_elem(elem: Self, n: usize, mc: &Mutation<'gc>) -> Vec<'gc, Self> {
        let mut v = Vec::with_capacity(mc, n);
        v.extend_with(mc, n, elem);
        v
    }
}

impl<'gc> SpecFromElem<'gc> for i8 {
    #[inline]
    #[track_caller]
    fn from_elem(elem: Self, n: usize, mc: &Mutation<'gc>) -> Vec<'gc, Self> {
        if elem == 0 {
            return Vec {
                buf: Gc::new_zeroed_slice(mc, n),
                len: n,
            };
        }
        let mut v = Vec::with_capacity(mc, n);
        unsafe {
            ptr::write_bytes(v.as_mut_ptr(), elem as u8, n);
            v.set_len(n);
        }
        v
    }
}

impl<'gc> SpecFromElem<'gc> for u8 {
    #[inline]
    #[track_caller]
    fn from_elem(elem: Self, n: usize, mc: &Mutation<'gc>) -> Vec<'gc, Self> {
        if elem == 0 {
            return Vec {
                buf: Gc::new_zeroed_slice(mc, n),
                len: n,
            };
        }
        let mut v = Vec::with_capacity(mc, n);
        unsafe {
            ptr::write_bytes(v.as_mut_ptr(), elem, n);
            v.set_len(n);
        }
        v
    }
}

impl<'gc> SpecFromElem<'gc> for () {
    #[inline]
    fn from_elem(_: (), n: usize, mc: &Mutation<'gc>) -> Vec<'gc, Self> {
        let mut v = Vec::with_capacity(mc, n);
        // SAFETY: the capacity has just been set to `n`
        // and `()` is a ZST with trivial `Clone` implementation
        unsafe { v.set_len(n) };
        v
    }
}
