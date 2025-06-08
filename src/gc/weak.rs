use core::{
    fmt::{self, Debug},
    ptr::NonNull,
};

use crate::{
    Mutation,
    collect::{Collect, Trace},
    context::Finalization,
    gc::Gc,
    ptr::{Metadata, PtrMeta, PtrMetadata},
};

#[repr(transparent)]
pub struct Weak<'gc, T: ?Sized + 'gc, M: 'gc = PtrMeta<T>> {
    pub(crate) inner: Gc<'gc, T, M>,
}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Copy for Weak<'gc, T, M> {}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Clone for Weak<'gc, T, M> {
    #[inline]
    fn clone(&self) -> Weak<'gc, T, M> {
        *self
    }
}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Debug for Weak<'gc, T, M> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "(GC'd Weak)")
    }
}

unsafe impl<'gc, T: ?Sized + 'gc, M: 'gc> Collect<'gc> for Weak<'gc, T, M> {
    #[inline]
    fn trace<C: Trace<'gc>>(&mut self, cc: &mut C) {
        cc.trace_weak(self)
    }
}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Weak<'gc, T, M> {
    /// If the `Weak` pointer can be safely upgraded to a strong pointer,
    /// upgrade it.
    ///
    /// This will fail if the value the `Weak` points to is dropped, or if we
    /// are in the [`crate::arena::CollectionPhase::Sweeping`] phase and we
    /// know the pointer *will* be dropped.
    #[inline]
    pub fn upgrade(self, mc: &Mutation<'gc>) -> Option<Gc<'gc, T, M>> {
        mc.upgrade(self.inner.ptr).then_some(self.inner)
    }

    /// Returns whether the value referenced by this `Weak` has already been
    /// dropped.
    ///
    /// # Note
    ///
    /// This is not the same as using [`Weak::upgrade`] and checking if the
    /// result is `None`! A `Weak` pointer can fail to upgrade *without*
    /// having been dropped if the current collection
    /// phase is [`crate::arena::CollectionPhase::Sweeping`] and the pointer
    /// *will* be dropped.
    ///
    /// It is not safe to use this to use this and casting as a substitute for
    /// [`Weak::upgrade`].
    #[inline]
    pub fn is_dropped(self) -> bool {
        !self.inner.ptr.header().is_live()
    }
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: Metadata<'a, T>> Weak<'gc, T, M> {
    /// Returns true when a pointer is *dead* during finalization.
    ///
    /// This is a weaker condition than being *dropped*, as the pointer *may*
    /// still be valid. Being *dead* means that there were no strong pointers
    /// pointing to this weak pointer that were found by the marking phase,
    /// and if it is not already dropped, it *will* be dropped as soon as
    /// collection resumes.
    ///
    /// If the pointer is still valid, it may be resurrected using
    /// `Weak::upgrade` or `Weak::resurrect`.
    ///
    /// NOTE: This returns true if the pointer was destined to be collected at
    /// the **start** of the current finalization callback. Resurrecting one
    /// pointer can transitively resurrect others, and this method does not
    /// reflect this from within the same finalization call! If transitive
    /// resurrection is important, you may have to carefully call finalize
    /// multiple times for one collection cycle with marking stages in-between,
    /// and in the precise order that you want.
    #[inline]
    pub fn is_dead(self, fc: &Finalization<'gc>) -> bool {
        Gc::is_dead(fc, self.inner)
    }

    /// Manually marks a dead (but non-dropped) `Weak` as strongly reachable and
    /// keeps it alive.
    ///
    /// This is similar to a write barrier in that it moves the collection phase
    /// back to `Marking` if it is not already there. All transitively held
    /// pointers from this will also be marked as reachable once marking
    /// resumes.
    ///
    /// Returns the upgraded `Gc` pointer as a convenience. Whether or not the
    /// strong pointer is stored anywhere, the value and all transitively
    /// reachable values are still guaranteed to not be dropped this
    /// collection cycle.
    #[inline]
    pub fn resurrect(self, fc: &Finalization<'gc>) -> Option<Gc<'gc, T, M>> {
        // SAFETY: We know that we are currently marking, so any non-dropped pointer is
        // safe to resurrect.
        if self.inner.ptr.header().is_live() {
            Gc::resurrect(fc, self.inner);
            Some(self.inner)
        } else {
            None
        }
    }

    /// Returns true if two `Weak`s point to the same allocation.
    ///
    /// Similarly to `Rc::ptr_eq` and `Arc::ptr_eq`, this function ignores the
    /// metadata of `dyn` pointers.
    #[inline]
    pub fn ptr_eq(this: Self, other: Self) -> bool {
        this.addr() == other.addr()
    }

    #[inline]
    pub fn addr(self) -> NonNull<()> {
        Gc::addr(self.inner)
    }

    #[inline]
    pub fn metadata(self) -> M {
        Gc::metadata(self.inner)
    }

    #[inline]
    pub fn to_raw_parts(self) -> (NonNull<()>, M) {
        Gc::to_raw_parts(self.inner)
    }

    /// Retrieve a `Weak` from a raw pointer obtained from `Weak::as_raw`
    ///
    /// # Safety
    ///
    /// The provided pointer must have been obtained from `Weak::as_raw` or
    /// `Gc::as_raw`, and the pointer must not have been *fully* collected
    /// yet (it may be a dropped but valid weak pointer).
    #[inline]
    pub unsafe fn from_raw(raw: NonNull<()>) -> Weak<'gc, T, M> {
        // SAFETY: The caller guarantees that this is safe.
        Weak {
            inner: unsafe { Gc::from_raw(raw) },
        }
    }
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: PtrMetadata<'a, T>> Weak<'gc, T, M> {
    #[inline]
    pub fn as_ptr(self) -> *const T {
        Gc::as_ptr(self.inner)
    }

    /// Retrieve a `Weak` from a raw pointer obtained from `Weak::as_ptr`
    ///
    /// # Safety
    ///
    /// The provided pointer must have been obtained from [`Weak::as_ptr`] or
    /// [`Gc::as_ptr`], and the pointer must not have been *fully* collected
    /// yet (it may be a dropped but valid weak pointer).
    #[inline]
    pub unsafe fn from_ptr(ptr: *const T) -> Weak<'gc, T, M> {
        // SAFETY: The caller guarantees that this is safe.
        Weak {
            inner: unsafe { Gc::from_ptr(ptr) },
        }
    }
}

impl<'gc, T: 'gc + ?Sized, M: 'gc> Weak<'gc, T, M> {
    /// Cast the internal pointer to a different type.
    ///
    /// # Safety
    ///
    /// It must be valid to dereference a `*mut U` that has come from casting a
    /// `*mut T`.
    #[inline]
    pub unsafe fn cast<U: 'gc, N: 'gc>(this: Weak<'gc, T, M>) -> Weak<'gc, U, N> {
        // SAFETY: The caller guarantees that this is safe.
        Weak {
            inner: unsafe { Gc::cast::<U, N>(this.inner) },
        }
    }
}
