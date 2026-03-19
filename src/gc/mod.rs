use core::{
    any::Any,
    borrow::Borrow,
    convert::Infallible,
    error::Error,
    fmt::{self, Debug, Display, Pointer},
    hash::{Hash, Hasher},
    marker::{PhantomData, Unsize},
    mem::MaybeUninit,
    ops::Deref,
    ptr::NonNull,
    str::Utf8Error,
};

use crate::{
    Finalization,
    barrier::{Unlock, Write},
    collect::{Collect, Static, Trace},
    context::Mutation,
    ptr::{MetaCollect, Metadata, PtrMeta, PtrMetadata, Uninit, native},
    types::{GcBox, GcColor, Invariant},
    vec::Vec,
};

mod unique;
mod weak;

pub use self::{unique::Unique, weak::Weak};

/// A garbage collected pointer to a type T.
///
/// Implements Copy, and is implemented as a plain machine pointer. You can only
/// allocate `Gc` pointers through a [`&Mutation<'gc>`] inside an arena type,
/// and through "generativity" such `Gc` pointers may not escape the arena they
/// were born in or be stored inside TLS. This, combined with correct `Collect`
/// implementations, means that `Gc` pointers will never be dangling and are
/// always safe to access.
///
/// # Layout
///
/// The underlying pointer points directly to the value, so it can be safety
/// [transmute]d when the type is [`Sized`]. However, since it is a **thin**
/// pointer, it cannot be safely cast to the corresponding raw pointer when the
/// type is `?Sized`. To obtain a raw pointer, use the [`Gc::as_ptr`] method, or
/// [`Gc::to_raw_parts`] if the metadata is customed.
///
/// [transmute]: core::mem::transmute
/// [`&Mutation<'gc>`]: crate::context::Mutation
#[repr(transparent)]
pub struct Gc<'gc, T: ?Sized + 'gc, M: 'gc = PtrMeta<T>> {
    pub(crate) ptr: GcBox,
    pub(crate) _invariant: Invariant<'gc, T, M>,
}

impl<'gc, 'a, T: Debug, M> Debug for Gc<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

impl<'gc, 'a, T: ?Sized + 'gc + 'a, M: Metadata<'a, T>> Pointer for Gc<'gc, T, M> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&Gc::addr(*self), fmt)
    }
}

impl<'gc, 'a, T: Display, M> Display for Gc<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Copy for Gc<'gc, T, M> {}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Clone for Gc<'gc, T, M> {
    #[inline]
    fn clone(&self) -> Gc<'gc, T, M> {
        *self
    }
}

unsafe impl<'gc, T: ?Sized + 'gc, M: 'gc> Collect<'gc> for Gc<'gc, T, M> {
    #[inline]
    fn trace<C: Trace<'gc>>(&mut self, cc: &mut C) {
        cc.trace_gc(self)
    }
}

impl<'gc, 'a, T, M> Deref for Gc<'gc, T, M>
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

impl<'gc, 'a, T, M> AsRef<T> for Gc<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'gc, 'a, T, M> Borrow<T> for Gc<'gc, T, M>
where
    T: ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T>,
{
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

impl<'gc, T: 'gc + Uninit, M: 'gc> Gc<'gc, T, M> {
    /// Creates a new uninitialized unique `Gc` with its associated metadata.
    pub fn with_metadata<'a, const ZEROED: bool>(mc: &Mutation<'gc>, meta: M) -> Unique<'gc, T, M>
    where
        T: 'a,
        M: MetaCollect<'gc, 'a, T>,
    {
        Unique::with_metadata::<ZEROED>(mc, meta)
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Gc<'gc, T> {
    /// Create a new `Gc` pointer from a sized value.
    #[inline]
    pub fn new(mc: &Mutation<'gc>, t: T) -> Gc<'gc, T> {
        Unique::new(mc, t).into_gc()
    }

    /// Create a new unique `Gc` pointer from a sized value.
    #[inline]
    pub fn unique(mc: &Mutation<'gc>, t: T) -> Unique<'gc, T> {
        Unique::new(mc, t)
    }

    /// Creates a new `Gc` pointer containing the given value, unsizing
    /// to a dynamically sized type.
    pub fn new_unsize<'a, Dyn>(mc: &Mutation<'gc>, t: T) -> Gc<'gc, Dyn>
    where
        T: Unsize<Dyn> + 'a,
        Dyn: 'gc + 'a + ?Sized,
        native::Unsized<Dyn>: MetaCollect<'gc, 'a, T>,
        PtrMeta<Dyn>: Metadata<'a, Dyn>,
    {
        Unique::new_unsize::<Dyn>(mc, t).into_gc()
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Gc<'gc, T> {
    /// Create a new uninit `Gc` pointer.
    #[inline]
    pub fn new_uninit(mc: &Mutation<'gc>) -> Unique<'gc, MaybeUninit<T>> {
        Unique::new_uninit(mc)
    }

    /// Create a new zeroed `Gc` pointer.
    #[inline]
    pub fn new_zeroed(mc: &Mutation<'gc>) -> Unique<'gc, MaybeUninit<T>> {
        Unique::new_zeroed(mc)
    }
}

impl<'gc, T: Collect<'gc> + 'gc> Gc<'gc, T> {
    /// Create a new uninit `Gc` pointer slice.
    pub fn new_uninit_slice(mc: &Mutation<'gc>, len: usize) -> Unique<'gc, [MaybeUninit<T>]> {
        Unique::new_uninit_slice(mc, len)
    }

    /// Create a new zeroed `Gc` pointer slice.
    pub fn new_zeroed_slice(mc: &Mutation<'gc>, len: usize) -> Unique<'gc, [MaybeUninit<T>]> {
        Unique::new_zeroed_slice(mc, len)
    }

    /// Transforms an iterator into a `Gc<'gc, [T]>`.
    ///
    /// The signature of this function differs from [`Iterator::collect`] from
    /// the standard library since a [`Mutation`] is required to handle the
    /// allocation.
    pub fn collect<I: IntoIterator<Item = T>>(mc: &Mutation<'gc>, iter: I) -> Gc<'gc, [T]> {
        Unique::collect(mc, iter).into_gc()
    }
}

impl<'gc, T: 'static> Gc<'gc, T> {
    /// Create a new `Gc` pointer from a static value.
    ///
    /// This method does not require that the type `T` implement `Collect`. This
    /// uses [`Static`] internally to automatically provide a trivial
    /// `Collect` impl and is equivalent to the following code:
    ///
    /// ```rust
    /// # use gc_arena::{Gc, collect::Static};
    /// # fn main() {
    /// # gc_arena::arena::rootless_mutate(|mc| {
    /// struct MyStaticStruct;
    /// let p = Gc::new(mc, Static(MyStaticStruct));
    /// // This is allowed because `Static` is `#[repr(transparent)]`
    /// let p: Gc<MyStaticStruct> = unsafe { Gc::cast(p) };
    /// # });
    /// # }
    /// ```
    #[inline]
    pub fn new_static(mc: &Mutation<'gc>, t: T) -> Gc<'gc, T> {
        let p = Gc::new(mc, Static(t));
        // SAFETY: `Static` is `#[repr(transparent)]`.
        unsafe { Gc::cast::<T, _>(p) }
    }
}

impl<'gc> Gc<'gc, dyn Any> {
    /// Cast a `Gc` pointer to a concrete type.
    pub fn downcast<T: Any>(self) -> Option<Gc<'gc, T>> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Gc<dyn Any>`, so it is valid to cast to `T`.
            Some(unsafe { Gc::cast::<T, _>(self) })
        } else {
            None
        }
    }

    /// Cast a `Gc` pointer to a concrete type without checks.
    ///
    /// # Safety
    ///
    /// `self` must be a `Gc<T>`.
    pub unsafe fn downcast_unchecked<T: Any>(self) -> Gc<'gc, T> {
        // SAFETY: `self` is a `Gc<dyn Any>`, so it is valid to cast to `T`.
        unsafe { Gc::cast::<T, _>(self) }
    }
}

impl<'gc> Gc<'gc, dyn Error + 'static> {
    /// Cast a `Gc` pointer to a concrete type.
    pub fn downcast<T: Error + 'static>(self) -> Option<Gc<'gc, T>> {
        if self.is::<T>() {
            // SAFETY: `self` is a `Gc<dyn Error>`, so it is valid to cast to `T`.
            Some(unsafe { Gc::cast::<T, _>(self) })
        } else {
            None
        }
    }

    /// Cast a `Gc` pointer to a concrete type.
    ///
    /// # Safety
    ///
    /// `self` must contains a `Unique<T>`.
    pub unsafe fn downcast_unchecked<T: Error + 'static>(self) -> Gc<'gc, T> {
        // SAFETY: `self` is a `Gc<dyn Any>`, so it is valid to cast to `T`.
        unsafe { Gc::cast::<T, _>(self) }
    }
}

impl<'gc, T: ?Sized + 'gc, M: 'gc> Gc<'gc, T, M> {
    /// Cast a `Gc` pointer to a different type.
    ///
    /// # Safety
    ///
    /// It must be valid to dereference a `N<U>::Ptr` that has come from casting
    /// a `M<T>::Ptr`.
    #[inline]
    pub unsafe fn cast<U: 'gc + ?Sized, N: 'gc>(this: Gc<'gc, T, M>) -> Gc<'gc, U, N> {
        Gc {
            ptr: this.ptr,
            _invariant: PhantomData,
        }
    }
}

impl<'gc, T: 'gc + ?Sized, M: Metadata<'gc, T>> Gc<'gc, T, M> {
    /// Obtains a long-lived reference to the contents of this `Gc`.
    ///
    /// Unlike `AsRef` or `Deref`, the returned reference isn't bound to the
    /// `Gc` itself, and will stay valid for the entirety of the current
    /// arena callback.
    #[inline]
    pub fn get_ref(this: Self) -> M::Ref {
        // SAFETY: The returned reference cannot escape the current arena callback, as
        // `&'gc T` never implements `Collect` (unless `'gc` is `'static`, which is
        // impossible here), and so cannot be stored inside the GC root.
        unsafe { M::as_ref(this.ptr.unerase::<T, M>()) }
    }
}

impl<'gc, T: 'gc + ?Sized, M: PtrMetadata<'gc, T>> Gc<'gc, T, M> {
    /// Triggers a write barrier on this `Gc`, allowing for safe mutation.
    ///
    /// This triggers an unrestricted *backwards* write barrier on this pointer,
    /// meaning that it is guaranteed that this pointer can safely adopt
    /// *any* arbitrary child pointers (until the next time that collection
    /// is triggered).
    ///
    /// It returns a reference to the inner `T` wrapped in a `Write` marker to
    /// allow for unrestricted mutation on the held type or any of its
    /// directly held fields.
    #[inline]
    pub fn write(mc: &Mutation<'gc>, gc: Self) -> &'gc Write<T> {
        unsafe {
            mc.backward_barrier::<_, Infallible, _, Infallible>(gc, None);
            // SAFETY: the write barrier stays valid until the end of the current callback.
            Write::assume(Self::get_ref(gc))
        }
    }

    /// Shorthand for [`Gc::write`]`(mc, self).`[`unlock()`](Write::unlock).
    #[inline]
    pub fn unlock(self, mc: &Mutation<'gc>) -> &'gc T::Unlocked
    where
        T: Unlock,
    {
        Gc::write(mc, self).unlock()
    }
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: PtrMetadata<'a, T>> Gc<'gc, T, M> {
    /// Returns a raw pointer to the `Gc`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is
    /// properly aligned, and points to a valid instance of `T`
    pub fn as_ptr(this: Self) -> *const T {
        unsafe { this.ptr.unerase::<T, M>().as_ptr() }
    }

    /// Constructs a `Gc` from a raw pointer.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`Gc::as_ptr`] within the
    /// same mutation session.
    pub unsafe fn from_ptr(raw: *const T) -> Self {
        Gc {
            // SAFETY: `raw` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::erase::<T, M>(NonNull::new_unchecked(raw.cast_mut())) },
            _invariant: PhantomData,
        }
    }
}

impl<'gc, 'a, T: 'gc + 'a + ?Sized, M: Metadata<'a, T>> Gc<'gc, T, M> {
    /// Obtains a [weaked] version of the `Gc` pointer. Useful for breaking
    /// reference cycles and clarify ownership relations.
    ///
    /// [weaked]: Weak
    #[inline]
    pub const fn downgrade(this: Gc<'gc, T, M>) -> Weak<'gc, T, M> {
        Weak { inner: this }
    }

    /// Returns the raw pointer parts to the `Gc`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is
    /// properly aligned, and points to a valid instance of `T`
    pub const fn addr(this: Self) -> NonNull<()> {
        this.ptr.into_raw()
    }

    /// Returns the metadata associated with this `Gc` pointer.
    ///
    /// Similar to [`core::ptr::metadata`].
    pub const fn metadata(this: Self) -> M {
        unsafe { this.ptr.metadata::<M>() }
    }

    /// Returns the raw pointer parts to the `Gc`'s contents.
    ///
    /// Very few guarantees are given about this pointer, except that it is
    /// properly aligned, and points to a valid instance of `T`
    pub const fn to_raw_parts(this: Self) -> (NonNull<()>, M) {
        (Self::addr(this), Self::metadata(this))
    }

    /// Constructs a `Gc` from a raw address.
    ///
    /// # Safety
    ///
    /// The given pointer must have been obtained from [`Gc::addr`] or
    /// [`Gc::to_raw_parts`] within the same mutation session.
    pub const unsafe fn from_raw(raw: NonNull<()>) -> Self {
        Gc {
            // SAFETY: `raw` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::from_raw(raw) },
            _invariant: PhantomData,
        }
    }

    /// Converts the `Gc` pointer into a `Unique` pointer, assuming that it is
    /// the only reference to the allocation.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that this is the only reference to the
    /// underlying allocation.
    #[inline]
    pub const unsafe fn as_unique_unchecked(this: Self) -> Unique<'gc, T, M> {
        // SAFETY: `this` is guaranteed to be unique.
        unsafe { Unique::from_raw(Gc::addr(this)) }
    }

    /// Returns true if two `Gc`s point to the same allocation.
    ///
    /// Similarly to `Rc::ptr_eq` and `Arc::ptr_eq`, this function ignores the
    /// metadata of `dyn` pointers.
    #[inline]
    pub fn ptr_eq(this: Self, other: Self) -> bool {
        Gc::addr(this) == Gc::addr(other)
    }

    /// Returns true when a pointer is *dead* during finalization. This is
    /// equivalent to `Weak::is_dead` for strong pointers.
    ///
    /// Any strong pointer reachable from the root will never be dead, BUT there
    /// can be strong pointers reachable only through other weak pointers
    /// that can be dead.
    #[inline]
    pub fn is_dead(_: &Finalization<'gc>, gc: Self) -> bool {
        matches!(gc.ptr.header().color(), GcColor::White | GcColor::WhiteWeak)
    }

    /// Manually marks a dead `Gc` pointer as reachable and keeps it alive.
    ///
    /// Equivalent to `Weak::resurrect` for strong pointers. Manually marks this
    /// pointer and all transitively held pointers as reachable, thus
    /// keeping them from being dropped this collection cycle.
    #[inline]
    pub fn resurrect(fc: &Finalization<'gc>, gc: Self) {
        fc.resurrect(gc.ptr);
    }
}

impl<'gc> Gc<'gc, str> {
    /// Converts the `Gc` byte slice into a GC'd string slice.
    ///
    /// # Errors
    ///
    /// If the slice is not valid UTF-8, an error will be returned.
    pub fn from_utf8(this: Gc<'gc, [u8]>) -> Result<Self, Utf8Error> {
        match str::from_utf8(&this) {
            // SAFETY: `this` is guaranteed to be valid UTF-8.
            Ok(_) => Ok(unsafe { Self::from_utf8_unchecked(this) }),
            Err(e) => Err(e),
        }
    }

    /// Converts the `Gc` byte slice into a GC'd string slice without checks.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that this contains a valid UTF-8 string.
    pub const unsafe fn from_utf8_unchecked(this: Gc<'gc, [u8]>) -> Self {
        let (ptr, _) = Gc::to_raw_parts(this);
        // SAFETY: The caller guarantees that this is a valid UTF-8 string.
        unsafe { Gc::from_raw(ptr) }
    }
}

impl<'gc, T: 'gc> IntoIterator for Gc<'gc, [T]> {
    type Item = &'gc T;

    type IntoIter = core::slice::Iter<'gc, T>;

    fn into_iter(self) -> Self::IntoIter {
        Gc::get_ref(self).iter()
    }
}

impl<'gc, 'a, 'b, T, U, M, N> PartialEq<Gc<'gc, U, N>> for Gc<'gc, T, M>
where
    T: PartialEq<U> + ?Sized + 'gc + 'a,
    U: ?Sized + 'gc + 'b,
    M: PtrMetadata<'a, T> + 'gc,
    N: PtrMetadata<'b, U> + 'gc,
{
    fn eq(&self, other: &Gc<'gc, U, N>) -> bool {
        (**self).eq(other)
    }
}

impl<'gc, 'a, T, M> Eq for Gc<'gc, T, M>
where
    T: Eq + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
}

impl<'gc, 'a, 'b, T, U, M, N> PartialOrd<Gc<'gc, U, N>> for Gc<'gc, T, M>
where
    T: PartialOrd<U> + ?Sized + 'gc + 'a,
    U: ?Sized + 'gc + 'b,
    M: PtrMetadata<'a, T> + 'gc,
    N: PtrMetadata<'b, U> + 'gc,
{
    fn partial_cmp(&self, other: &Gc<'gc, U, N>) -> Option<core::cmp::Ordering> {
        (**self).partial_cmp(other)
    }

    fn le(&self, other: &Gc<'gc, U, N>) -> bool {
        (**self).le(other)
    }

    fn lt(&self, other: &Gc<'gc, U, N>) -> bool {
        (**self).lt(other)
    }

    fn ge(&self, other: &Gc<'gc, U, N>) -> bool {
        (**self).ge(other)
    }

    fn gt(&self, other: &Gc<'gc, U, N>) -> bool {
        (**self).gt(other)
    }
}

impl<'gc, 'a, T, M> Ord for Gc<'gc, T, M>
where
    T: Ord + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (**self).cmp(other)
    }
}

impl<'gc, 'a, T, M> Hash for Gc<'gc, T, M>
where
    T: Hash + ?Sized + 'gc + 'a,
    M: PtrMetadata<'a, T> + 'gc,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<'gc, T: 'gc + Collect<'gc>> From<(&Mutation<'gc>, Vec<'gc, T>)> for Gc<'gc, [T]> {
    fn from((mc, vec): (&Mutation<'gc>, Vec<'gc, T>)) -> Self {
        vec.into_gc_slice(mc)
    }
}

impl<'gc, T: 'gc + Collect<'gc>> From<(Vec<'gc, T>, &Mutation<'gc>)> for Gc<'gc, [T]> {
    fn from((vec, mc): (Vec<'gc, T>, &Mutation<'gc>)) -> Self {
        vec.into_gc_slice(mc)
    }
}
