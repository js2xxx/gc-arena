use core::{
    alloc::Layout,
    any::Any,
    borrow::Borrow,
    error::Error,
    fmt::{self, Debug, Display, Pointer},
    marker::{PhantomData, Unsize},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::{NonNull, Pointee},
};

use crate::{
    Gc,
    collect::{Collect, Trace},
    context::Mutation,
    types::{GcBox, GcBoxHeader, GcBoxInner, Invariant, MetaLayout},
    vec::Vec,
};

/// A uniquely-owned garbage-collected pointer to a type `T`.
///
/// Unlike [`Gc`] this pointer is known to be unique, and as such allows mutation
/// without the use of interior mutability.
///
/// [`Gc`]: crate::Gc
/// [`Collect`]: crate::Collect
#[repr(transparent)]
pub struct Unique<'gc, T: ?Sized + 'gc> {
    ptr: GcBox,
    _invariant: Invariant<'gc, T>,
}

impl<'gc, T: Debug + ?Sized + 'gc> Debug for Unique<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc> Pointer for Unique<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&Unique::as_ptr(self), fmt)
    }
}

impl<'gc, T: Display + ?Sized + 'gc> Display for Unique<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, fmt)
    }
}

unsafe impl<'gc, T: Collect<'gc> + ?Sized + 'gc> Collect<'gc> for Unique<'gc, T> {
    const NEEDS_TRACE: bool = true;

    fn trace<U: Trace<'gc>>(&self, cc: &mut U) {
        // SAFETY: The constructed `Gc` pointer won't access the underlying value;
        // the dropping process will not start either until the next collection phase after
        // the current `Unique` is dropped.
        cc.trace_gc(unsafe { Gc::from_ptr(self.ptr.unerased_value::<()>()) });
    }
}

impl<'gc, T: ?Sized + 'gc> Deref for Unique<'gc, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.ptr.unerased_value::<T>() }
    }
}

impl<'gc, T: ?Sized + 'gc> DerefMut for Unique<'gc, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.ptr.unerased_value::<T>() }
    }
}

impl<'gc, T: ?Sized + 'gc> AsRef<T> for Unique<'gc, T> {
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'gc, T: ?Sized + 'gc> AsMut<T> for Unique<'gc, T> {
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<'gc, T: ?Sized + 'gc> Borrow<T> for Unique<'gc, T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, T> {
    /// Creates a new `Unique` containing the given value.
    ///
    /// As the allocated value is statically known not to have any other references, we can safely
    /// modify it through this pointer.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let mut gc = Unique::new(mc, 42i32);
    ///
    /// assert_eq!(*gc, 42);
    /// *gc = 0;
    /// assert_eq!(*gc, 0);
    /// # });
    /// ```
    #[inline]
    pub fn new(mc: &Mutation<'gc>, t: T) -> Unique<'gc, T> {
        // The following code is equivalent to:
        //
        //     Unique::write(Unique::new_uninit(mc), t)
        //
        // The shorthand is used here to avoid an extra assignment to
        // the underlying VTable.

        let ptr = mc.allocate::<T, false>(());
        // SAFETY: `ptr` is a valid uninit pointer to `T`.
        unsafe { ptr.unerased_value::<T>().write(t) };
        Unique {
            ptr,
            _invariant: PhantomData,
        }
    }

    /// Creates a new `Unique` containing the given value, unsizing to a dynamically sized type.
    #[inline]
    pub fn new_unsize<Dyn>(mc: &Mutation<'gc>, t: T) -> Unique<'gc, Dyn>
    where
        T: Collect<'gc> + Unsize<Dyn>,
        Dyn: ?Sized + 'gc,
        <Dyn as Pointee>::Metadata: MetaLayout<Dyn>,
    {
        let ptr = core::ptr::from_ref(&t) as *const Dyn;
        let (_, metadata) = ptr.to_raw_parts();

        let gc_box = mc.allocate_unsize::<T, Dyn, false>(metadata);
        // SAFETY: `ptr` is a uninit pointer to `Dyn` which can receive a `T`.
        unsafe { gc_box.unerased_value::<T>().write(t) };

        Unique {
            ptr: gc_box,
            _invariant: PhantomData,
        }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, T> {
    /// Creates a new `Unique` with uninitialized contents.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let mut gc = Unique::<i32>::new_uninit(mc);
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
    pub fn new_uninit(mc: &Mutation<'gc>) -> Unique<'gc, MaybeUninit<T>> {
        Unique {
            ptr: mc.allocate::<MaybeUninit<T>, false>(()),
            _invariant: PhantomData,
        }
    }

    /// Creates a new `Unique` with uninitialized contents, with the memory being filled with `0` bytes.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let gc = Unique::<i32>::new_zeroed(mc);
    ///
    /// // SAFETY: It is valid to initialize an `i32` with zeroed bytes.
    /// let gc = unsafe { gc.assume_init() };
    ///
    /// assert_eq!(*gc, 0);
    /// # });
    /// ```
    #[inline]
    pub fn new_zeroed(mc: &Mutation<'gc>) -> Unique<'gc, MaybeUninit<T>> {
        let ret = Unique {
            ptr: mc.allocate::<MaybeUninit<T>, true>(()),
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

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, [T]> {
    /// Constructs a new garbage-collected slice with uninitialized contents.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let mut values = Unique::<[i32]>::new_uninit_slice(mc, 3);
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
    pub fn new_uninit_slice(mc: &Mutation<'gc>, len: usize) -> Unique<'gc, [MaybeUninit<T>]> {
        Unique {
            ptr: mc.allocate::<[MaybeUninit<T>], false>(len),
            _invariant: PhantomData,
        }
    }

    /// Constructs a new garbage-collected slice with unitialized contents, with the memory being filled with `0` bytes.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let values = Unique::<[i32]>::new_zeroed_slice(mc, 3);
    /// let values = unsafe { values.assume_init() };
    ///
    /// assert_eq!(*values, [0, 0, 0]);
    /// # });
    /// ```
    pub fn new_zeroed_slice(mc: &Mutation<'gc>, len: usize) -> Unique<'gc, [MaybeUninit<T>]> {
        let ret: Unique<'gc, [MaybeUninit<T>]> = Unique {
            ptr: mc.allocate::<[MaybeUninit<T>], true>(len),
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

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, MaybeUninit<T>> {
    /// Converts to `Unique<'gc, T>`.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to guarantee
    /// that the value really is in an initialized state. Calling this when the
    /// content is not yet fully initialized will likely cause undefined behaviour.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let mut gc = Unique::<i32>::new_uninit(mc);
    ///
    /// let gc: Unique<'_, i32> = unsafe {
    ///     gc.as_mut_ptr().write(42);
    ///
    ///     gc.assume_init()
    /// };
    ///
    /// assert_eq!(*gc, 42);
    /// # });
    /// ```
    #[inline]
    pub unsafe fn assume_init(self) -> Unique<'gc, T> {
        // SAFETY: The caller guarantees that the value is initialized.
        unsafe { self.ptr.header().reset_vtable::<T>() };
        Unique {
            ptr: self.ptr,
            _invariant: PhantomData,
        }
    }

    /// Writes the value and converts to `Unique<'gc, T>`
    ///
    /// This method converts the pointer similarly to [`Unique::assume_init`]
    /// but writes `value` into it before conversion, thus guaranteeing safety.
    pub fn write(mut this: Self, value: T) -> Unique<'gc, T> {
        (*this).write(value);
        // SAFETY: The value is initialized by `value`.
        unsafe { Self::assume_init(this) }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, [MaybeUninit<T>]> {
    /// Converts to `Unique<'gc, [T]>`.
    ///
    /// # Safety
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to
    /// guarantee that the values really are in an initialized state. Calling
    /// this when the contents are not yet fully initialized will likely cause
    /// undefined behaviour.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let mut values = Unique::<[i32]>::new_uninit_slice(mc, 3);
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
    pub unsafe fn assume_init(self) -> Unique<'gc, [T]> {
        // SAFETY: The caller guarantees that the slice is initialized.
        unsafe { self.ptr.header().reset_vtable::<[T]>() };
        Unique {
            ptr: self.ptr,
            _invariant: PhantomData,
        }
    }

    /// Constructs a new garbage-collected slice, cloning each element from the given slice.
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// use std::rc::Rc;
    ///
    /// let src = [Rc::new(1), Rc::new(2), Rc::new(3), Rc::new(4)];
    ///
    /// let gc = Unique::new_uninit_slice(mc, 2);
    /// let gc = Unique::write_clone_of_slice(gc, &src[1..3]);
    ///
    /// assert_eq!(src.map(|rc| Rc::strong_count(&rc)), [1, 2, 2, 1]);
    /// # });
    /// ```
    pub fn write_clone_of_slice(mut this: Self, src: &[T]) -> Unique<'gc, [T]>
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
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let src = [1, 2, 3, 4];
    ///
    /// let gc = Unique::new_uninit_slice(mc, 2);
    /// let gc = Unique::write_copy_of_slice(gc, &src[1..3]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(*gc, [2, 3]);
    /// # });
    /// ```
    ///
    /// [`write_clone_of_slice`]: Self::write_clone_of_slice
    pub fn write_copy_of_slice(mut this: Self, src: &[T]) -> Unique<'gc, [T]>
    where
        T: Copy,
    {
        (*this).write_copy_of_slice(src);
        // SAFETY: The value is initialized by `src`.
        unsafe { Self::assume_init(this) }
    }
}

impl<'gc> Unique<'gc, dyn Any> {
    pub fn downcast<T: Any>(self) -> Result<Unique<'gc, T>, Self> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Unique<dyn Any>`, so it is valid to cast to `T`.
            Ok(unsafe { Unique::cast(self) })
        } else {
            Err(self)
        }
    }
}

impl<'gc> Unique<'gc, dyn Error + 'static> {
    pub fn downcast<T: Error + 'static>(self) -> Result<Unique<'gc, T>, Self> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Unique<dyn Error>`, so it is valid to cast to `T`.
            Ok(unsafe { Unique::cast(self) })
        } else {
            Err(self)
        }
    }
}

impl<'gc, T: ?Sized + 'gc> Unique<'gc, T> {
    /// Cast a `Unique` GC pointer to a different type.
    ///
    /// # Safety
    ///
    /// It must be valid to dereference a `*mut U` that has come from casting a `*mut T`.
    #[inline]
    pub unsafe fn cast<U: 'gc>(this: Unique<'gc, T>) -> Unique<'gc, U> {
        Unique {
            ptr: this.ptr,
            _invariant: PhantomData,
        }
    }

    /// Returns a raw mutable pointer to the `Unique`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, points to a valid instance of `T`, and may be written to.
    pub fn as_mut_ptr(this: &mut Unique<'gc, T>) -> *mut T {
        // SAFETY: `Unique` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Returns a raw pointer to the `Unique`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, and points to a valid instance of `T`
    pub fn as_ptr(this: &Unique<'gc, T>) -> *const T {
        // SAFETY: `Unique` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Transforms the `Unique` into a raw pointer.
    ///
    /// The pointer is guaranteed to be valid only in the current collection phase.
    pub fn into_raw(this: Unique<'gc, T>) -> *mut T {
        // SAFETY: `Unique` is guaranteed to contain a pointer to a valid instance of a `GcBoxInner<T>`.
        unsafe { this.ptr.unerased_value::<T>() }
    }

    /// Constructs a `Unique` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`Unique::as_ptr`] or
    /// [`Gc::as_ptr`]. There must also exist no other garbage collected pointers
    /// which point to the same allocation. This is always the case for [`Unique::as_ptr`].
    pub unsafe fn from_raw(raw: *mut T) -> Unique<'gc, T> {
        let layout = Layout::new::<GcBoxHeader>();
        // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
        let (_, header_offset) = layout
            .extend(unsafe { Layout::for_value_raw(raw) })
            .unwrap();
        // SAFETY: `ptr` is previously obtained from `Gc::as_ptr`, so there is always a header.
        let ptr = unsafe { raw.byte_sub(header_offset) } as *mut GcBoxInner<T>;
        Unique {
            // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::erase(NonNull::new_unchecked(ptr)) },
            _invariant: PhantomData,
        }
    }

    /// Converts the `Unique` into a regular [`Gc`].
    pub fn into_gc(self) -> Gc<'gc, T> {
        // SAFETY: Trivial.
        unsafe { Gc::from_ptr(Unique::into_raw(self)) }
    }
}

impl<'gc, T: 'gc> Unique<'gc, [T]> {
    pub fn into_vec(self) -> Vec<'gc, T> {
        let () = Vec::<'gc, T>::ASSERT_NO_DROP;
        // SAFETY: the elements is handled separately in `Vec`s, so
        // assign the VTable to uninitalized states.
        unsafe { self.ptr.header().reset_vtable::<[MaybeUninit<T>]>() };

        let (ptr, len) = Self::into_raw(self).to_raw_parts();
        // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
        unsafe { Vec::from_raw_parts(ptr.cast(), len, len) }
    }
}

impl<'gc, T: 'gc + Collect<'gc>> From<(&Mutation<'gc>, Vec<'gc, T>)> for Unique<'gc, [T]> {
    fn from((mc, vec): (&Mutation<'gc>, Vec<'gc, T>)) -> Self {
        vec.into_unique_slice(mc)
    }
}

impl<'gc, T: 'gc + Collect<'gc>> From<(Vec<'gc, T>, &Mutation<'gc>)> for Unique<'gc, [T]> {
    fn from((vec, mc): (Vec<'gc, T>, &Mutation<'gc>)) -> Self {
        vec.into_unique_slice(mc)
    }
}

impl<'gc, T: ?Sized + 'gc> From<Unique<'gc, T>> for Gc<'gc, T> {
    fn from(value: Unique<'gc, T>) -> Self {
        value.into_gc()
    }
}

#[cfg(test)]
mod test {
    use std::string::ToString;

    use super::*;
    use crate::{Collect, arena::rootless_mutate};

    #[test]
    fn unique_gc_drops() {
        use std::{cell::Cell, thread_local};
        thread_local! {
            static DROPPED: Cell<usize> = const { Cell::new(0) };
        }

        struct DropWatcher;

        impl Drop for DropWatcher {
            fn drop(&mut self) {
                DROPPED.set(DROPPED.get() + 1);
            }
        }

        // SAFETY: DropWatcher's drop implementation does not dereference any garbage-collected pointers.
        unsafe impl Collect<'_> for DropWatcher {
            const NEEDS_TRACE: bool = false;
        }

        rootless_mutate(|mc| {
            Unique::new(mc, DropWatcher);
            Unique::write(Unique::new_uninit(mc), DropWatcher);
            Unique::<DropWatcher>::new_uninit(mc);
        });

        assert_eq!(DROPPED.get(), 2);
    }

    #[test]
    fn unique_gc_new() {
        rootless_mutate(|mc| {
            let mut gc = Unique::new(mc, "hello".to_string());
            assert_eq!(*gc, "hello");
            *gc = "world".to_string();
            assert_eq!(*gc, "world");

            let mut uninit = Unique::write(Unique::new_uninit(mc), "hello".to_string());
            assert_eq!(*uninit, "hello");
            *uninit = "world".to_string();
            assert_eq!(*uninit, "world");

            assert_eq!(gc.ptr.header().vtable(), uninit.ptr.header().vtable());
        });
    }

    #[test]
    fn unique_gc_uninit() {
        rootless_mutate(|mc| {
            let gc = Unique::new_uninit(mc);
            let gc1 = Unique::write(gc, 0);
            assert_eq!(*gc1, 0);

            // SAFETY: `i32` can be safely zero-initialized.
            let gc2 = unsafe { Unique::<i32>::new_zeroed(mc).assume_init() };
            assert_eq!(*gc1, *gc2);
        });
    }
}
