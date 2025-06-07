use core::{
    any::Any,
    borrow::{Borrow, BorrowMut},
    error::Error,
    fmt::{self, Debug, Display, Pointer},
    hash::{Hash, Hasher},
    marker::{PhantomData, Unsize},
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

use crate::{
    collect::{Collect, Trace}, context::Mutation, ptr::{native, MetaCollect, Metadata, PtrMeta, PtrMetadata, Uninit}, types::{GcBox, GcColor, Invariant}, vec::Vec, Finalization, Gc
};

/// A uniquely-owned garbage-collected pointer to a type `T`.
///
/// Unlike [`Gc`] this pointer is known to be unique, and as such allows mutation
/// without the use of interior mutability.
///
/// [`Gc`]: crate::Gc
/// [`Collect`]: crate::Collect
#[repr(transparent)]
pub struct Unique<'gc, T: ?Sized + 'gc, M: 'gc = PtrMeta<T>> {
    pub(crate) ptr: GcBox,
    _invariant: Invariant<'gc, T, M>,
}

impl<'gc, 'a, T: Debug, M> Debug for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

impl<'gc, 'a, T: ?Sized + 'gc + 'a, M: Metadata<'a, T>> Pointer for Unique<'gc, T, M> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&Unique::addr(self), fmt)
    }
}

impl<'gc, 'a, T: Display, M> Display for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, fmt)
    }
}

unsafe impl<'gc, T: ?Sized + 'gc, M: 'gc> Collect<'gc> for Unique<'gc, T, M> {
    fn trace<U: Trace<'gc>>(&self, cc: &mut U) {
        cc.trace_unique(self);
    }
}

impl<'gc, 'a, T, M> Deref for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { self.ptr.unerase::<T, M>().as_ref() }
    }
}

impl<'gc, 'a, T, M> DerefMut for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    #[inline]
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.ptr.unerase::<T, M>().as_mut() }
    }
}

impl<'gc, 'a, T, M> AsRef<T> for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'gc, 'a, T, M> AsMut<T> for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn as_mut(&mut self) -> &mut T {
        self
    }
}

impl<'gc, 'a, T, M> Borrow<T> for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn borrow(&self) -> &T {
        self
    }
}

impl<'gc, 'a, T, M> BorrowMut<T> for Unique<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn borrow_mut(&mut self) -> &mut T {
        self
    }
}

impl<'gc, T: ?Sized + Uninit + 'gc, M: 'gc> Unique<'gc, T, M> {
    /// Creates a new uninitialized `Unique` with its associated metadata.
    pub fn with_metadata<'a, const ZEROED: bool>(mc: &Mutation<'gc>, meta: M) -> Unique<'gc, T, M>
    where
        T: 'a,
        M: MetaCollect<'gc, 'a, T>,
    {
        let ptr = mc.allocate::<T, M, ZEROED>(meta);
        let ret: Unique<'_, T, M> = Unique {
            ptr,
            _invariant: PhantomData,
        };
        #[cfg(not(miri))]
        return ret;
        #[cfg(miri)]
        {
            let mut ret = ret;
            let addr = Unique::addr_mut(&mut ret);
            // SAFETY: The metadata is valid for this type.
            let size = unsafe { ret.ptr.metadata::<M>().layout().unwrap_unchecked().size() };
            // SAFETY: The memory is uninitialized and valid.
            unsafe { core::ptr::write_bytes::<u8>(addr.as_ptr().cast(), 0, size) };
            ret
        }
    }

    /// Converts to `Unique<'gc, T::Init, M>`, assuming its contents are initialized.
    ///
    /// # Safety
    ///
    /// As with [`MaybeUninit::assume_init`], it is up to the caller to guarantee
    /// that the value really is in an initialized state. Calling this when the
    /// content is not yet fully initialized will likely cause undefined behaviour.
    ///
    /// # Examples
    ///
    /// For sized types:
    ///
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
    ///
    /// For slices:
    ///
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
    pub unsafe fn assume_init<'a>(self) -> Unique<'gc, T::Init, M>
    where
        T::Init: 'a,
        M: MetaCollect<'gc, 'a, T::Init>,
    {
        unsafe { self.ptr.header().reset_vtable::<T::Init, M>() };
        let needs_trace = unsafe { self.ptr.metadata::<M>().needs_trace() };
        self.ptr.header().set_needs_trace(needs_trace);
        Unique {
            ptr: self.ptr,
            _invariant: PhantomData,
        }
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

        let ptr = mc.allocate::<T, (), false>(());
        // SAFETY: `ptr` is a valid uninit pointer to `T`.
        unsafe { ptr.unerase::<T, ()>().write(t) };
        Unique {
            ptr,
            _invariant: PhantomData,
        }
    }

    /// Creates a new `Unique` containing the given value, unsizing to a dynamically sized type.
    #[inline]
    pub fn new_unsize<'a, Dyn>(mc: &Mutation<'gc>, t: T) -> Unique<'gc, Dyn>
    where
        T: Unsize<Dyn> + 'a,
        Dyn: 'gc + ?Sized,
        native::Unsized<Dyn>: MetaCollect<'gc, 'a, T, Ptr = NonNull<T>>,
    {
        let metadata = core::ptr::metadata(&t as &Dyn);
        let gc_box = mc.allocate::<T, _, false>(native::Unsized(metadata));
        // SAFETY: `ptr` is a uninit pointer to `Dyn` which can receive a `T`.
        unsafe { gc_box.unerase::<T, ()>().write(t) };

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
        Unique::with_metadata::<false>(mc, ())
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
        Unique::with_metadata::<true>(mc, ())
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
        Unique::with_metadata::<false>(mc, len)
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
        Unique::with_metadata::<true>(mc, len)
    }

    /// Transforms an iterator into a `Unique<'gc, [T]>`.
    ///
    /// The signature of this function differs from [`Iterator::collect`] from the
    /// standard library since a [`Mutation`] is required to handle the allocation.
    pub fn collect<I: IntoIterator<Item = T>>(mc: &Mutation<'gc>, iter: I) -> Unique<'gc, [T]> {
        Vec::collect(mc, iter).into_unique_slice(mc)
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, MaybeUninit<T>> {
    /// Writes the value and converts to `Unique<'gc, T>`
    ///
    /// This method converts the pointer similarly to [`Unique::assume_init`]
    /// but writes `value` into it before conversion, thus guaranteeing safety.
    pub fn write(mut this: Self, value: T) -> Unique<'gc, T> {
        (*this).write(value);
        // SAFETY: The value is initialized by `value`.
        unsafe { this.assume_init() }
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Unique<'gc, [MaybeUninit<T>]> {
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
    /// let gc = gc.write_clone_of_slice(&src[1..3]);
    ///
    /// assert_eq!(src.map(|rc| Rc::strong_count(&rc)), [1, 2, 2, 1]);
    /// # });
    /// ```
    pub fn write_clone_of_slice(mut self, src: &[T]) -> Unique<'gc, [T]>
    where
        T: Clone,
    {
        (*self).write_clone_of_slice(src);
        // SAFETY: The value is initialized by `src`.
        unsafe { self.assume_init() }
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
    /// let gc = gc.write_copy_of_slice(&src[1..3]);
    ///
    /// assert_eq!(src, [1, 2, 3, 4]);
    /// assert_eq!(*gc, [2, 3]);
    /// # });
    /// ```
    ///
    /// [`write_clone_of_slice`]: Self::write_clone_of_slice
    pub fn write_copy_of_slice(mut self, src: &[T]) -> Unique<'gc, [T]>
    where
        T: Copy,
    {
        (*self).write_copy_of_slice(src);
        // SAFETY: The value is initialized by `src`.
        unsafe { Self::assume_init(self) }
    }

    /// Constructs a new garbage-collected slice, initializing each element with the given value.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let gc = Unique::new_uninit_slice(mc, 3);
    /// let gc = gc.write_filled(42);
    ///
    /// assert_eq!(*gc, [42, 42, 42]);
    /// # });
    /// ```
    pub fn write_filled(mut self, value: T) -> Unique<'gc, [T]>
    where
        T: Clone,
    {
        (*self).write_filled(value);
        // SAFETY: The value is initialized by `value`.
        unsafe { Self::assume_init(self) }
    }

    /// Constructs a new garbage-collected slice, initializing each element with the given function.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, gc::Unique};
    /// # rootless_mutate(|mc| {
    /// let gc = Unique::new_uninit_slice(mc, 3);
    /// let gc = gc.write_with(|i| i * 2);
    ///
    /// assert_eq!(*gc, [0, 2, 4]);
    /// # });
    /// ```
    pub fn write_with<F>(mut self, f: F) -> Unique<'gc, [T]>
    where
        F: FnMut(usize) -> T,
    {
        (*self).write_with(f);
        // SAFETY: The value is initialized by `f`.
        unsafe { Self::assume_init(self) }
    }
}

impl<'gc> Unique<'gc, dyn Any> {
    /// Cast a `Unique` GC pointer to a concrete type.
    pub fn downcast<T: Any>(self) -> Result<Unique<'gc, T>, Self> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Unique<dyn Any>`, so it is valid to cast to `T`.
            Ok(unsafe { Unique::cast(self) })
        } else {
            Err(self)
        }
    }

    /// Cast a `Unique` GC pointer to a concrete type without checks.
    ///
    /// # Safety
    ///
    /// `self` must be a `Unique<T>`.
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Unique<'gc, T> {
        // SAFETY: `self` is a `Unique<dyn Any>`, so it is valid to cast to `T`.
        unsafe { Unique::cast(self) }
    }
}

impl<'gc> Unique<'gc, dyn Error + 'static> {
    /// Cast a `Unique` GC pointer to a concrete type.
    pub fn downcast<T: Error + 'static>(self) -> Result<Unique<'gc, T>, Self> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Unique<dyn Error>`, so it is valid to cast to `T`.
            Ok(unsafe { Unique::cast(self) })
        } else {
            Err(self)
        }
    }

    /// Cast a `Unique` GC pointer to a concrete type without checks.
    ///
    /// # Safety
    ///
    /// `self` must be a `Unique<T>`.
    pub unsafe fn downcast_unchecked<T: Error + 'static>(self) -> Unique<'gc, T> {
        // SAFETY: `self` is a `Unique<dyn Error>`, so it is valid to cast to `T`.
        unsafe { Unique::cast(self) }
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
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: Metadata<'a, T>> Unique<'gc, T, M> {
    /// Returns the raw address of the `Unique`.
    pub fn addr(this: &Self) -> NonNull<()> {
        this.ptr.into_raw()
    }

    /// Returns the raw address of the `Unique`.
    ///
    /// The mutable variant of [`Unique::addr`] exists for clarity of borrowing.
    pub fn addr_mut(this: &mut Self) -> NonNull<()> {
        this.ptr.into_raw()
    }

    /// Returns the metadata associated with the `Unique`.
    pub fn metadata(this: &Self) -> M {
        unsafe { this.ptr.metadata::<M>() }
    }

    /// Transforms the `Unique` into a raw pointer.
    ///
    /// The pointer is guaranteed to be valid only in the current collection phase.
    pub fn into_raw_parts(this: Self) -> (NonNull<()>, M) {
        (Self::addr(&this), Self::metadata(&this))
    }

    /// Constructs a `Unique` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`Unique::addr`],
    /// [`Unique::into_raw_parts`], or [`Gc::addr`] within the same mutation session.
    /// There must also exist no other garbage collected pointers which point to the
    /// same allocation.
    pub unsafe fn from_raw(raw: NonNull<()>) -> Unique<'gc, T, M> {
        Unique {
            // SAFETY: `raw` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::from_raw(raw) },
            _invariant: PhantomData,
        }
    }

    /// Converts the `Unique` into a regular [`Gc`].
    pub fn into_gc(self) -> Gc<'gc, T, M> {
        // SAFETY: Trivial.
        unsafe { Gc::from_raw(Unique::into_raw_parts(self).0) }
    }

    /// Returns true when a pointer is *dead* during finalization. This is equivalent to
    /// [`Gc::is_dead`].
    ///
    /// Any unique pointer reachable from the root will never be dead.
    #[inline]
    pub fn is_dead(_: &Finalization<'gc>, gc: &Self) -> bool {
        matches!(gc.ptr.header().color(), GcColor::White | GcColor::WhiteWeak)
    }

    /// Manually marks a dead `Unique` GC pointer as reachable and keeps it alive.
    ///
    /// Equivalent to [`Gc::resurrect`]. Manually marks this pointer and all transitively
    /// held pointers as reachable, thus keeping them from being dropped this collection
    /// cycle.
    #[inline]
    pub fn resurrect(fc: &Finalization<'gc>, gc: &Self) {
        fc.resurrect(gc.ptr);
    }
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: PtrMetadata<'a, T>> Unique<'gc, T, M> {
    /// Returns a raw pointer to the `Unique`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, and points to a valid instance of `T`
    pub fn as_ptr(this: &Self) -> *const T {
        unsafe { this.ptr.unerase::<T, M>().as_ptr() }
    }

    /// Returns a raw mutable pointer to the `Unique`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is properly
    /// aligned, points to a valid instance of `T`, and may be written to.
    pub fn as_mut_ptr(this: &mut Self) -> *mut T {
        unsafe { this.ptr.unerase::<T, M>().as_ptr() }
    }

    /// Transforms the `Unique` into a raw pointer.
    ///
    /// The pointer is guaranteed to be valid only in the current collection phase.
    pub fn into_ptr(this: Self) -> *mut T {
        unsafe { this.ptr.unerase::<T, M>().as_ptr() }
    }

    /// Constructs a `Unique` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`Unique::as_ptr`], [`Unique::into_ptr`],
    /// or [`Gc::as_ptr`] within the same mutation session. There must also exist no other
    /// garbage collected pointers which point to the same allocation.
    pub unsafe fn from_ptr(raw: *mut T) -> Unique<'gc, T, M> {
        Unique {
            // SAFETY: `raw` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::erase::<T, M>(NonNull::new_unchecked(raw)) },
            _invariant: PhantomData,
        }
    }
}

impl<'gc, T: 'gc> Unique<'gc, [T]> {
    /// Converts the `Unique` into a GC'd [`Vec`].
    pub fn into_vec(self) -> Vec<'gc, T> {
        let () = Vec::<'gc, T>::ASSERT_NO_DROP;
        // SAFETY: the elements is handled separately in `Vec`s, so
        // assign the VTable to uninitalized states.
        unsafe { self.ptr.header().reset_vtable::<[MaybeUninit<T>], usize>() };
        self.ptr.header().set_needs_trace(false);

        let (ptr, len) = Unique::into_ptr(self).to_raw_parts();
        // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
        unsafe { Vec::from_raw_parts(ptr.cast(), len, len) }
    }
}

impl<'gc, T: 'gc> IntoIterator for Unique<'gc, [T]> {
    type Item = T;

    type IntoIter = crate::vec::IntoIter<'gc, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, 'gc, T: 'gc> IntoIterator for &'a Unique<'gc, [T]> {
    type Item = &'a T;

    type IntoIter = core::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, 'gc, T: 'gc> IntoIterator for &'a mut Unique<'gc, [T]> {
    type Item = &'a mut T;

    type IntoIter = core::slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'gc, 'a, 'b, T, U, M, N> PartialEq<Unique<'gc, U, N>> for Unique<'gc, T, M>
where
    T: PartialEq<U> + ?Sized + 'gc + 'a,
    U: ?Sized + 'gc + 'b,
    M: PtrMetadata<'a, T> + 'gc,
    N: PtrMetadata<'b, U> + 'gc,
{
    fn eq(&self, other: &Unique<'gc, U, N>) -> bool {
        (**self).eq(other)
    }
}

impl<'gc, 'a, T, M> Eq for Unique<'gc, T, M>
where
    T: Eq + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
}

impl<'gc, 'a, 'b, T, U, M, N> PartialOrd<Unique<'gc, U, N>> for Unique<'gc, T, M>
where
    T: PartialOrd<U> + ?Sized + 'gc + 'a,
    U: ?Sized + 'gc + 'b,
    M: PtrMetadata<'a, T> + 'gc,
    N: PtrMetadata<'b, U> + 'gc,
{
    fn partial_cmp(&self, other: &Unique<'gc, U, N>) -> Option<core::cmp::Ordering> {
        (**self).partial_cmp(other)
    }

    fn le(&self, other: &Unique<'gc, U, N>) -> bool {
        (**self).le(other)
    }

    fn lt(&self, other: &Unique<'gc, U, N>) -> bool {
        (**self).lt(other)
    }

    fn ge(&self, other: &Unique<'gc, U, N>) -> bool {
        (**self).ge(other)
    }

    fn gt(&self, other: &Unique<'gc, U, N>) -> bool {
        (**self).gt(other)
    }
}

impl<'gc, 'a, T, M> Ord for Unique<'gc, T, M>
where
    T: Ord + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (**self).cmp(other)
    }
}

impl<'gc, 'a, T, M> Hash for Unique<'gc, T, M>
where
    T: Hash + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
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

impl<'gc, 'a, T, M> From<Unique<'gc, T, M>> for Gc<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: Metadata<'a, T> + 'gc,
{
    fn from(value: Unique<'gc, T, M>) -> Self {
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
