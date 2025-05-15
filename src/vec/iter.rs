use core::{
    fmt,
    iter::{FusedIterator, TrustedLen},
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
    slice,
};

use crate::{Collect, collect::Trace, gc::Unique};

use super::Vec;

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

impl<'gc, T: 'gc> IntoIter<'gc, T> {
    pub(super) fn from_vec(vec: Vec<'gc, T>) -> Self {
        let this = ManuallyDrop::new(vec);
        // SAFETY: `this.buf` is valid for `this.len` elements of type `T`.
        unsafe {
            let mut buf = ptr::read(&this.buf);
            let start = NonNull::new_unchecked(buf.as_mut_ptr().cast::<T>());
            let end = start.add(this.len);
            Self { buf, start, end }
        }
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `self.start` and `self.end` are valid
        // pointers to the same allocation as `self.buf`.
        unsafe { slice::from_raw_parts(self.start.as_ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `self.start` and `self.end` are valid
        // pointers to the same allocation as `self.buf`.
        unsafe { slice::from_raw_parts_mut(self.start.as_ptr(), self.len()) }
    }

    pub(crate) fn forget_remaining(&mut self) {
        // For the ZST case, it is crucial that we mutate `end` here, not `ptr`.
        // `ptr` must stay aligned, while `end` may be unaligned.
        self.end = self.start;
    }

    fn as_raw_mut_slice(&mut self) -> *mut [T] {
        ptr::from_raw_parts_mut(self.start.as_ptr(), self.len())
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

impl<'gc, T: 'gc> Drop for IntoIter<'gc, T> {
    fn drop(&mut self) {
        // SAFETY: `self.start..self.end` contains valid
        // elements of the same allocation as `self.buf`.
        unsafe { ptr::drop_in_place(self.as_raw_mut_slice()) };
        // GC handles deallocation.
    }
}
