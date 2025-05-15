use core::{
    alloc::Layout,
    borrow::Borrow,
    fmt::{self, Debug, Display, Pointer},
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    Gc,
    collect::Collect,
    context::Mutation,
    types::{GcBox, GcBoxHeader, GcBoxInner, Invariant},
};

/// A uniquely-owned garbage-collected pointer to a type `T`.
///
/// Unlike [`Gc`] this pointer is known to be unique,
/// and as such allows mutation without the use of interior mutability. It does not however,
/// implement [`Collect`], and as such is intended for the initialisation of data, before
/// converting into a plain [`Gc`] with [`UniqueGc::into_gc`].
///
/// [`Gc`]: crate::Gc
/// [`Collect`]: crate::Collect
#[repr(transparent)]
pub struct UniqueGc<'gc, T: ?Sized + 'gc> {
    ptr: GcBox,
    _invariant: Invariant<'gc, T>,
}

impl<'gc, T: Debug + ?Sized + 'gc> Debug for UniqueGc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc> Pointer for UniqueGc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&UniqueGc::as_ptr(self), fmt)
    }
}

impl<'gc, T: Display + ?Sized + 'gc> Display for UniqueGc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc> Deref for UniqueGc<'gc, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.ptr.unerased_value::<T>() }
    }
}

impl<'gc, T: ?Sized + 'gc> DerefMut for UniqueGc<'gc, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.unerased_value::<T>() }
    }
}

impl<'gc, T: ?Sized + 'gc> AsRef<T> for UniqueGc<'gc, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'gc, T: ?Sized + 'gc> AsMut<T> for UniqueGc<'gc, T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<'gc, T: ?Sized + 'gc> Borrow<T> for UniqueGc<'gc, T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<'gc, T: Collect<'gc> + 'gc> UniqueGc<'gc, T> {
    /// Creates a new `UniqueGc` containing the given value.
    ///
    /// As the allocated value is statically known not to have any other references, we can safely
    /// modify it through this pointer.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let mut gc = UniqueGc::new(mc, 42i32);
    ///
    /// assert_eq!(*gc, 42);
    /// *gc = 0;
    /// assert_eq!(*gc, 0);
    /// # });
    /// ```
    #[inline]
    pub fn new(mc: &Mutation<'gc>, t: T) -> UniqueGc<'gc, T> {
        UniqueGc::write(UniqueGc::new_uninit(mc), t)
    }
}

impl<'gc, T: Collect<'gc> + 'gc> UniqueGc<'gc, T> {
    /// Creates a new `UniqueGc` with uninitialized contents.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let mut gc = UniqueGc::<i32>::new_uninit(mc);
    ///
    /// // let gc = unsafe { gc.assume_init() };
    /// //                      ^ undefined behaviour
    ///
    /// (*gc).write(42);
    /// let gc = unsafe { gc.assume_init() };
    /// assert_eq!(*gc, 42);
    /// # })
    /// ```
    #[inline]
    pub fn new_uninit(mc: &Mutation<'gc>) -> UniqueGc<'gc, MaybeUninit<T>> {
        UniqueGc {
            ptr: mc.allocate::<T, false>(()),
            _invariant: PhantomData,
        }
    }

    /// Creates a new `UniqueGc` with uninitialized contents, with the memory being filled with `0` bytes.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let gc = UniqueGc::<i32>::new_zeroed(mc);
    ///
    /// // SAFETY: It is valid to initialize an `i32` with zeroed bytes.
    /// let gc = unsafe { gc.assume_init() };
    ///
    /// assert_eq!(*gc, 0);
    /// # });
    /// ```
    #[inline]
    pub fn new_zeroed(mc: &Mutation<'gc>) -> UniqueGc<'gc, MaybeUninit<T>> {
        let ret = UniqueGc {
            ptr: mc.allocate::<T, true>(()),
            _invariant: PhantomData,
        };
        #[cfg(not(miri))]
        return ret;
        #[cfg(miri)]
        {
            // Miri doesn't support zeroing, so we need to do it manually.
            let mut ret = ret;
            core::slice::from_mut(&mut *ret)
                .as_bytes_mut()
                .fill(MaybeUninit::new(0));
            return ret;
        }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> UniqueGc<'gc, [T]> {
    /// Constructs a new garbage-collected slice with uninitialized contents.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let mut values = UniqueGc::<[i32]>::new_uninit_slice(mc, 3);
    ///
    /// let values = unsafe {
    ///     // Deferred initialization:
    ///     values[0].as_mut_ptr().write(1);
    ///     values[1].as_mut_ptr().write(2);
    ///     values[2].as_mut_ptr().write(3);
    ///
    ///     values.assume_init()
    /// };
    ///
    /// assert_eq!(*values, [1, 2, 3]);
    /// # });
    /// ```
    pub fn new_uninit_slice(mc: &Mutation<'gc>, len: usize) -> UniqueGc<'gc, [MaybeUninit<T>]> {
        UniqueGc {
            ptr: mc.allocate::<[T], false>(len),
            _invariant: PhantomData,
        }
    }

    /// Constructs a new garbage-collected slice with unitialized contents, with the memory being filled with `0` bytes.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let values = UniqueGc::<[i32]>::new_zeroed_slice(mc, 3);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0]);
    /// # });
    /// ```
    pub fn new_zeroed_slice(mc: &Mutation<'gc>, len: usize) -> UniqueGc<'gc, [MaybeUninit<T>]> {
        let ret: UniqueGc<'gc, [MaybeUninit<T>]> = UniqueGc {
            ptr: mc.allocate::<[T], true>(len),
            _invariant: PhantomData,
        };
        #[cfg(not(miri))]
        return ret;
        #[cfg(miri)]
        {
            // Miri doesn't support zeroing, so we need to do it manually.
            let mut ret = ret;
            ret.as_bytes_mut().fill(MaybeUninit::new(0));
            return ret;
        }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> UniqueGc<'gc, MaybeUninit<T>> {
    /// Converts to `UniqueGc<'gc, T>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to guarantee
    /// that the value really is in an initialized state. Calling this when the
    /// content is not yet fully initialized will likely cause undefined behaviour.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let mut gc = UniqueGc::<i32>::new_uninit(mc);
    ///
    /// let gc: UniqueGc<'_, i32> = unsafe {
    ///     gc.as_mut_ptr().write(42);
    ///
    ///     gc.assume_init()
    /// };
    ///
    /// assert_eq!(*gc, 42);
    /// # });
    /// ```
    #[inline]
    pub unsafe fn assume_init(self) -> UniqueGc<'gc, T> {
        UniqueGc {
            ptr: self.ptr,
            _invariant: PhantomData,
        }
    }

    /// Writes the value and converts to `UniqueGc<'gc, T>`
    ///
    /// This method converts the pointer similarly to [`UniqueGc::assume_init`]
    /// but writes `value` into it before conversion, thus guaranteeing safety.
    pub fn write(mut this: Self, value: T) -> UniqueGc<'gc, T> {
        (*this).write(value);
        // SAFETY: The value is initialized by `value`.
        unsafe { Self::assume_init(this) }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> UniqueGc<'gc, [MaybeUninit<T>]> {
    /// Converts to `UniqueGc<'gc, [T]>`.
    ///
    /// # Safety
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to
    /// guarantee that the values really are in an initialized state. Calling
    /// this when the contents are not yet fully initialized will likely cause
    /// undefined behaviour.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let mut values = UniqueGc::<[i32]>::new_uninit_slice(mc, 3);
    ///
    /// let values = unsafe {
    ///     // Deferred initialization:
    ///     values[0].as_mut_ptr().write(1);
    ///     values[1].as_mut_ptr().write(2);
    ///     values[2].as_mut_ptr().write(3);
    ///
    ///     values.assume_init()
    /// };
    ///
    /// assert_eq!(*values, [1, 2, 3]);
    /// # });
    /// ```
    #[inline]
    pub unsafe fn assume_init(self) -> UniqueGc<'gc, [T]> {
        UniqueGc {
            ptr: self.ptr,
            _invariant: PhantomData,
        }
    }

    /// Constructs a new garbage-collected slice, cloning each element from the given slice.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// use std::rc::Rc;
    ///
    /// let src = [Rc::new(1), Rc::new(2), Rc::new(3), Rc::new(4)];
    ///
    /// let gc = UniqueGc::new_uninit_slice(mc, 2);
    /// let gc = UniqueGc::write_clone_of_slice(gc, &src[1..3]);
    ///
    /// assert_eq!(src.map(|rc| Rc::strong_count(&rc)), [1, 2, 2, 1]);
    /// # });
    /// ```
    pub fn write_clone_of_slice(mut this: Self, src: &[T]) -> UniqueGc<'gc, [T]>
    where
        T: Clone,
    {
        (*this).write_clone_of_slice(src);
        // SAFETY: The value is initialized by `src`.
        unsafe { Self::assume_init(this) }
    }

    /// Constructs a new garbage-collected slice, copying each element from the given slice.
    ///
    /// If `T` does not implement `Copy`, use [`write_clone_of_slice`].
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::UniqueGc};
    /// # rootless_mutate(|mc| {
    /// let src = [1, 2, 3, 4];
    ///
    /// let gc = UniqueGc::new_uninit_slice(mc, 2);
    /// let gc = UniqueGc::write_copy_of_slice(gc, &src[1..3]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(*gc, [2, 3]);
    /// # });
    /// ```
    ///
    /// [`write_clone_of_slice`]: Self::write_clone_of_slice
    pub fn write_copy_of_slice(mut this: Self, src: &[T]) -> UniqueGc<'gc, [T]>
    where
        T: Copy,
    {
        (*this).write_copy_of_slice(src);
        // SAFETY: The value is initialized by `src`.
        unsafe { Self::assume_init(this) }
    }
}

impl<'gc, T: ?Sized + 'gc> UniqueGc<'gc, T> {
    /// Returns a raw mutable pointer to the `UniqueGc`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, points to a valid instance of `T`, and may be written to.
    pub fn as_mut_ptr(this: &mut UniqueGc<'gc, T>) -> *mut T {
        // SAFETY: `UniqueGc` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Returns a raw pointer to the `UniqueGc`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, and points to a valid instance of `T`
    pub fn as_ptr(this: &UniqueGc<'gc, T>) -> *const T {
        // SAFETY: `UniqueGc` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Transforms the `UniqueGc` into a raw pointer.
    ///
    /// The pointer is guaranteed to be valid only in the current collection phase.
    pub fn into_raw(this: UniqueGc<'gc, T>) -> *mut T {
        // SAFETY: `UniqueGc` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Constructs a `UniqueGc` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`UniqueGc::as_ptr`] or
    /// [`Gc::as_ptr`]. There must also exist no other garbage collected pointers
    /// which point to the same allocation. This is always the case for [`UniqueGc::as_ptr`].
    pub unsafe fn from_raw(raw: *mut T) -> UniqueGc<'gc, T> {
        let layout = Layout::new::<GcBoxHeader>();
        // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
        let (_, header_offset) = layout
            .extend(unsafe { Layout::for_value_raw(raw) })
            .unwrap();
        // SAFETY: `ptr` is previously obtained from `Gc::as_ptr`, so there is always a header.
        let ptr = unsafe { raw.byte_sub(header_offset) } as *mut GcBoxInner<T>;
        UniqueGc {
            // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::erase(NonNull::new_unchecked(ptr)) },
            _invariant: PhantomData,
        }
    }

    /// Converts the `UniqueGc` into a regular [`Gc`].
    pub fn into_gc(this: UniqueGc<'gc, T>) -> Gc<'gc, T> {
        // SAFETY: Trivial.
        unsafe { Gc::from_ptr(UniqueGc::into_raw(this)) }
    }
}

impl<'gc, T: ?Sized + 'gc> From<UniqueGc<'gc, T>> for Gc<'gc, T> {
    fn from(value: UniqueGc<'gc, T>) -> Self {
        UniqueGc::into_gc(value)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Collect, arena::rootless_mutate};

    #[test]
    fn unique_gc_drops() {
        use std::{cell::Cell, thread_local};
        thread_local! {
            static DROPPED: Cell<bool> = const { Cell::new(false) };
        }

        struct DropWatcher;

        impl Drop for DropWatcher {
            fn drop(&mut self) {
                DROPPED.set(true);
            }
        }

        // SAFETY: DropWatcher's drop implementation does not dereference any garbage-collected pointers.
        unsafe impl Collect<'_> for DropWatcher {
            const NEEDS_TRACE: bool = false;
        }

        rootless_mutate(|mc| {
            UniqueGc::new(mc, DropWatcher);
        });

        assert!(DROPPED.get());
    }

    #[test]
    fn unique_gc_new() {
        rootless_mutate(|mc| {
            let mut gc = UniqueGc::new(mc, 12i32);
            assert_eq!(*gc, 12);
            *gc = 42;
            assert_eq!(*gc, 42);
        });
    }

    #[test]
    fn unique_gc_uninit() {
        rootless_mutate(|mc| {
            let gc = UniqueGc::new_uninit(mc);
            let gc1 = UniqueGc::write(gc, 0);
            assert_eq!(*gc1, 0);

            // SAFETY: `i32` can be safely zero-initialized.
            let gc2 = unsafe { UniqueGc::<i32>::new_zeroed(mc).assume_init() };
            assert_eq!(*gc1, *gc2);
        });
    }
}
