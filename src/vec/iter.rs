use core::{
    fmt,
    iter::{FusedIterator, TrustedLen},
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
    slice,
};

use crate::{Collect, Mutation, collect::Trace, gc::Unique};

use super::{SpecExtend, Vec};

pub struct IntoIter<'gc, T: 'gc> {
    buf: Unique<'gc, [MaybeUninit<T>]>,
    start: NonNull<T>,
    end: NonNull<T>,
}

impl<'gc, T: 'gc + fmt::Debug> fmt::Debug for IntoIter<'gc, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}

unsafe impl<'gc, T: Collect<'gc> + 'gc> Collect<'gc> for IntoIter<'gc, T> {
    fn trace<U: Trace<'gc>>(&self, cc: &mut U) {
        cc.trace(&self.buf);
        cc.trace(self.as_slice());
    }
}

impl<'gc, T: 'gc + Collect<'gc>> IntoIter<'gc, T> {
    /// Collects the remaining items of this iterator into a `Vec`.
    ///
    /// This method is provided because the generic [`Vec::collect`] cannot specialize on the
    /// type of this iterator (which is branded by `'gc`).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use gc_arena::{arena::rootless_mutate, vec, vec::IntoIter};
    /// # rootless_mutate(|mc| {
    /// let mut iter = vec![mc => 1, 2, 3, 4, 5].into_iter();
    /// assert_eq!(iter.next(), Some(1));
    /// assert_eq!(iter.next(), Some(2));
    /// assert_eq!(iter.next_back(), Some(5));
    /// assert_eq!(iter.into_vec(mc), [3, 4]);
    /// # });
    pub fn into_vec(self, mc: &Mutation<'gc>) -> Vec<'gc, T> {
        // A common case is passing a vector into a function which immediately
        // re-collects into a vector. We can short circuit this if the IntoIter
        // has not been advanced at all.
        // When it has been advanced We can also reuse the memory and move the data to the front.
        // But we only do so when the resulting Vec wouldn't have more unused capacity
        // than creating it through the generic FromIterator implementation would. That limitation
        // is not strictly necessary as Vec's allocation behavior is intentionally unspecified.
        // But it is a conservative choice.
        if self.len() >= self.capacity() / 2 {
            let this = ManuallyDrop::new(self);
            // SAFETY: `self.start` and `self.end` are valid
            // pointers to the same allocation as `self.buf`.
            return unsafe {
                let (ptr, cap) = Unique::into_ptr(ptr::read(&this.buf)).to_raw_parts();
                let dst = ptr.cast::<T>();
                let src = this.start.as_ptr();

                if !ptr::addr_eq(src, dst) {
                    ptr::copy(src, dst, this.len());
                }
                Vec::from_raw_parts(dst, this.len(), cap)
            };
        }

        let mut vec = Vec::with_capacity(mc, self.len());
        // must delegate to spec_extend() since extend() itself delegates
        // to spec_from for empty Vecs
        <Vec<T> as SpecExtend<'gc, T, _>>::extend(&mut vec, mc, self);
        vec
    }
}

impl<'gc, T: 'gc> IntoIter<'gc, T> {
    pub(super) fn from_vec(vec: Vec<'gc, T>) -> Self {
        let this = ManuallyDrop::new(vec);
        // SAFETY: `this.buf` is valid for `this.len` elements of type `T`.
        unsafe {
            let mut buf = ptr::read(&this.buf);
            let start = NonNull::new_unchecked(buf.as_mut_ptr().cast::<T>());
            let end = if size_of::<T>() == 0 {
                start.byte_add(this.len)
            } else {
                start.add(this.len)
            };
            Self { buf, start, end }
        }
    }

    /// Returns the remaining items of this iterator as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.start` and `self.end` are valid
        // pointers to the same allocation as `self.buf`.
        unsafe { slice::from_raw_parts(self.start.as_ptr(), self.len()) }
    }

    /// Returns the remaining items of this iterator as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.start` and `self.end` are valid
        // pointers to the same allocation as `self.buf`.
        unsafe { slice::from_raw_parts_mut(self.start.as_ptr(), self.len()) }
    }

    fn capacity(&self) -> usize {
        self.buf.len()
    }

    pub(super) fn forget_remaining(&mut self) {
        // For the ZST case, it is crucial that we mutate `end` here, not `ptr`.
        // `ptr` must stay aligned, while `end` may be unaligned.
        self.end = self.start;
    }
}

impl<'gc, T: 'gc> AsRef<[T]> for IntoIter<'gc, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<'gc, T: 'gc> Iterator for IntoIter<'gc, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let ptr = if const { size_of::<T>() == 0 } {
            if self.start == self.end {
                return None;
            }
            // `ptr` has to stay where it is to remain aligned, so we reduce the length by 1 by
            // reducing the `end`.
            self.end = unsafe { self.end.byte_sub(1) };
            self.start
        } else {
            if self.start == self.end {
                return None;
            }
            let old = self.start;
            self.start = unsafe { old.add(1) };
            old
        };
        Some(unsafe { ptr.read() })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = if const { size_of::<T>() == 0 } {
            self.end.addr().get() - self.start.addr().get()
        } else {
            // SAFETY: `self.start` and `self.end` are valid
            // pointers to the same allocation as `self.buf`.
            unsafe { self.end.offset_from_unsigned(self.start) }
        };
        (len, Some(len))
    }

    fn count(self) -> usize {
        self.len()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
}

impl<'gc, T: 'gc> DoubleEndedIterator for IntoIter<'gc, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let ptr = if const { size_of::<T>() == 0 } {
            if self.start == self.end {
                return None;
            }
            // `ptr` has to stay where it is to remain aligned, so we reduce the length by 1 by
            // reducing the `end`.
            self.end = unsafe { self.end.byte_sub(1) };
            self.start
        } else {
            if self.start == self.end {
                return None;
            }
            self.end = unsafe { self.end.sub(1) };
            self.end
        };
        Some(unsafe { ptr.read() })
    }
}

impl<'gc, T: 'gc> ExactSizeIterator for IntoIter<'gc, T> {}

impl<'gc, T: 'gc> FusedIterator for IntoIter<'gc, T> {}

unsafe impl<'gc, T: 'gc> TrustedLen for IntoIter<'gc, T> {}
