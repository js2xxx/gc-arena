use core::{alloc::Allocator, marker::PhantomData, mem, ops::Deref};

use crate::{
    context::Context,
    gc::{Gc, Weak},
    metrics::Metrics,
    ptr::MetaCollect,
    types::{GcBox, Invariant},
};

/// Handle value given by arena callbacks during construction and mutation.
///
/// Allows allocating new `Gc` pointers and internally mutating values held by
/// `Gc` pointers.
pub struct Mutation<'gc> {
    context: &'gc Context,
    alloc: &'gc dyn Allocator,
    _invariant: Invariant<'gc>,
}

impl<'gc> Mutation<'gc> {
    pub(crate) unsafe fn new(context: &Context, alloc: &dyn Allocator) -> Self {
        // SAFETY: `Mutation` is a transparent wrapper around `Context`.
        unsafe {
            Mutation {
                context: mem::transmute::<&Context, &'gc Context>(context),
                alloc: mem::transmute::<&dyn Allocator, &'gc dyn Allocator>(alloc),
                _invariant: PhantomData,
            }
        }
    }

    #[inline]
    pub fn metrics(&self) -> &Metrics {
        self.context.metrics()
    }

    /// IF we are in the marking phase AND the `parent` pointer is colored black
    /// AND the `child` (if given) is colored white, then change the `parent`
    /// color to gray and enqueue it for tracing.
    ///
    /// This operation is known as a "backwards write barrier". Calling this
    /// method is one of the safe ways for the value in the `parent` pointer
    /// to use internal mutability to adopt the `child` pointer without
    /// invalidating the color invariant.
    ///
    /// If the `child` parameter is given, then calling this method ensures that
    /// the `parent` pointer may safely adopt the `child` pointer. If no `child`
    /// is given, then calling this method is more general, and it ensures that
    /// the `parent` pointer may adopt *any* child pointer(s) before collection
    /// is next triggered.
    #[inline]
    pub fn backward_barrier<T, U, M, N>(&self, parent: Gc<'gc, T, M>, child: Option<Gc<'gc, U, N>>)
    where
        T: ?Sized + 'gc,
        U: ?Sized + 'gc,
        M: 'gc,
        N: 'gc,
    {
        self.context
            .backward_barrier(parent.ptr, child.map(|p| p.ptr))
    }

    /// A version of [`Mutation::backward_barrier`] that allows adopting a
    /// [`Weak`] child.
    #[inline]
    pub fn backward_barrier_weak<T, U, M, N>(&self, parent: Gc<'gc, T, M>, child: Weak<'gc, U, N>)
    where
        T: ?Sized + 'gc,
        U: ?Sized + 'gc,
        M: 'gc,
        N: 'gc,
    {
        self.context
            .backward_barrier_weak(parent.ptr, child.inner.ptr)
    }

    /// IF we are in the marking phase AND the `parent` pointer (if given) is
    /// colored black, AND the `child` is colored white, then immediately
    /// change the `child` to gray and enqueue it for tracing.
    ///
    /// This operation is known as a "forwards write barrier". Calling this
    /// method is one of the safe ways for the value in the `parent` pointer
    /// to use internal mutability to adopt the `child` pointer without
    /// invalidating the color invariant.
    ///
    /// If the `parent` parameter is given, then calling this method ensures
    /// that the `parent` pointer may safely adopt the `child` pointer. If
    /// no `parent` is given, then calling this method is more general, and
    /// it ensures that the `child` pointer may be adopted by *any*
    /// parent pointer(s) before collection is next triggered.
    #[inline]
    pub fn forward_barrier<T, U, M, N>(&self, parent: Option<Gc<'gc, T, M>>, child: Gc<'gc, U, N>)
    where
        T: ?Sized + 'gc,
        U: ?Sized + 'gc,
        M: 'gc,
        N: 'gc,
    {
        self.context
            .forward_barrier(parent.map(|p| p.ptr), child.ptr)
    }

    /// A version of [`Mutation::forward_barrier`] that allows adopting a
    /// [`Weak`] child.
    #[inline]
    pub fn forward_barrier_weak<T, U, M, N>(
        &self,
        parent: Option<Gc<'gc, T, M>>,
        child: Weak<'gc, U, N>,
    ) where
        T: ?Sized + 'gc,
        U: ?Sized + 'gc,
        M: 'gc,
        N: 'gc,
    {
        self.context
            .forward_barrier_weak(parent.map(|p| p.ptr), child.inner.ptr)
    }

    #[inline]
    pub(crate) fn allocate<'a, T, M, const ZEROED: bool>(&self, metadata: M) -> GcBox
    where
        T: 'a + ?Sized,
        M: MetaCollect<'gc, 'a, T>,
    {
        self.context.allocate::<T, M, false>(metadata, self.alloc)
    }

    #[inline]
    pub(crate) fn upgrade(&self, gc_box: GcBox) -> bool {
        self.context.upgrade(gc_box)
    }
}

/// Handle value given to finalization callbacks in `MarkedArena`.
///
/// Derefs to `Mutation<'gc>` to allow for arbitrary mutation, but adds
/// additional powers to examine the state of the fully marked arena.
pub struct Finalization<'gc>(Mutation<'gc>);

impl<'gc> Deref for Finalization<'gc> {
    type Target = Mutation<'gc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'gc> Finalization<'gc> {
    pub(crate) unsafe fn new(context: &Context, alloc: &dyn Allocator) -> Self {
        Self(unsafe { Mutation::new(context, alloc) })
    }

    #[inline]
    pub(crate) fn resurrect(&self, gc_box: GcBox) {
        self.context.resurrect(gc_box)
    }
}
