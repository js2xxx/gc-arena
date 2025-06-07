use core::{
    alloc::{Layout, LayoutError},
    mem::MaybeUninit,
    ptr::{DynMetadata, NonNull, Pointee},
};

use crate::{Collect, collect::Trace};

pub(crate) type PtrMeta<T> = <T as Pointee>::Metadata;

/// A type that is valid when uninitalized.
///
/// This trait is implemented for all sized objects and slices.
///
/// # Safety
///
/// The type must be valid when any part of its data is uninitialized.
pub unsafe trait Uninit {
    /// The initialized form of the type.
    type Init: ?Sized;
}

unsafe impl<T> Uninit for MaybeUninit<T> {
    type Init = T;
}

unsafe impl<T> Uninit for [MaybeUninit<T>] {
    type Init = [T];
}

/// A generalized pointer metadata type.
///
/// Every [built-in pointee type] has an associated metadata type that contains
/// some of its runtime information: the layout, the v-table, etc.
///
/// However, the built-in pointee type is not customizable:
///
/// - For sized types, the metadata is `()`;
/// - For slices and `str`, the metadata is `usize`.
/// - For trait objects, the metadata is [`DynMetadata<Self>`].
///
/// This makes it impossible to create a pointer type associated with a custom
/// metadata type. For example, in garbage collection, a metadata for dynamic
/// `struct`s might contains the offset and layout of all its fields, which
/// cannot be represented by `()`, `usize`, or `DynMetadata<Self>` trivially.
///
/// This trait enables us to create a pointer type associated with a custom
/// metadata type.
///
/// # Safety
///
/// - The metadata type must be contains valid information of the pointee type
///   even if its associated pointer is not dereferencable.
/// - The pointer type must inherit the "pointer-like" behavior. For example,
///   it must be possible to be dereferenced if its metadata and address are
///   valid.
/// - The reference type must inherit the "reference-like" behavior, just like
///   the built-in `&T`.
///
/// [built-in pointee type]: core::ptr::Pointee
pub unsafe trait Metadata<
    'a,
    T: ?Sized + 'a + Pointee<Metadata = Marker>,
    Marker = <T as Pointee>::Metadata,
>: Copy + PartialEq
{
    /// The associated pointer type.
    type Ptr: Copy;

    /// The associated reference type.
    type Ref: Copy;

    /// Creates a pointer from the given address and metadata.
    ///
    /// This function is equivalent to [`NonNull::from_raw_parts`].
    fn with_addr(self, addr: NonNull<()>) -> Self::Ptr;

    /// Returns the address of the pointer.
    ///
    /// This function is equivalent to [`NonNull::as_ptr`].
    fn addr(ptr: Self::Ptr) -> NonNull<()>;

    /// Returns a shared reference to the pointed value.
    ///
    /// This function is equivalent to [`NonNull::as_ref`].
    ///
    /// # Safety
    ///
    /// The pointer must be [convertible to a reference], in the context
    /// of the inherited pointer-like equivalence.
    ///
    /// [convertible to a reference]:
    /// https://doc.rust-lang.org/nightly/core/ptr/index.html#pointer-to-reference-conversion
    unsafe fn as_ref(ptr: Self::Ptr) -> Self::Ref;
}

/// A trait that implies the [`Metadata`] trait for built-in pointer types.
///
/// This trait is implemented for all built-in pointer metadata types, e.g., `()`,
/// `usize`, and `DynMetadata<Self>`. Nevertheless, users may also implement
/// this trait for their own custom pointer metadata types to enable [dereferencing
/// GC pointers].
///
/// # Safety
///
/// 1. The `ptr_metadata` function must return a valid metadata of the pointee type.
///    That is, the metadata must be valid even if the pointer is not safe to dereference.
/// 2. If this trait is implemented, the `Metadata` trait must also be implemented conforming
///    to the native pointer equivalents:
///    - `Ptr` and `Ref` must be `NonNull<T>` and `&'a T` respectively;
///    - `with_addr`, `addr`, and `as_ref` methods must forward to the built-in equivalents.
///
/// Implementors may find it lengthy to write such implementation code by hand. The
/// [`macro@PtrMetadata`] derive macro is provided to get rid of this boilerplate. However,
/// The first safety requirement is **not automatically satisfied** by the derive macro
/// (as annotated with the options in the attribute), which implementors must guarantee by
/// themselves.
///
/// [dereferencing GC pointers]: crate::gc::Gc
pub unsafe trait PtrMetadata<
    'a,
    T: ?Sized + 'a + Pointee<Metadata = Marker>,
    Marker = <T as Pointee>::Metadata,
>: Metadata<'a, T, Marker, Ptr = NonNull<T>, Ref = &'a T>
{
    /// Returns a valid pointer metadata of the pointee type.
    fn ptr_metadata(self) -> PtrMeta<T>;
}

pub use gc_arena_derive::PtrMetadata;

macro_rules! impl_ptr_metadata {
    ($ty:ty) => {
        type Ptr = NonNull<$ty>;

        type Ref = &'a $ty;

        fn with_addr(self, addr: NonNull<()>) -> NonNull<$ty> {
            NonNull::<$ty>::from_raw_parts(addr, PtrMetadata::<'a, $ty>::ptr_metadata(self))
        }

        #[inline]
        fn addr(ptr: NonNull<$ty>) -> NonNull<()> {
            ptr.cast()
        }

        #[inline]
        unsafe fn as_ref(ptr: NonNull<$ty>) -> &'a $ty {
            unsafe { ptr.as_ref() }
        }
    };
}

/// A generalized pointer metadata type with layout information.
///
/// # Safety
///
/// - The `layout` function must return a valid layout of the pointee type.
/// - The `drop_in_place` function must drop the pointee type correctly, if
///   the provided pointer is valid.
pub unsafe trait MetaLayout<
    'a,
    T: ?Sized + 'a + Pointee<Metadata = Marker>,
    Marker = <T as Pointee>::Metadata,
>: Metadata<'a, T, Marker>
{
    /// Returns the layout of its associated pointee object.
    fn layout(self) -> Result<Layout, LayoutError>;

    /// Drops the pointee type at the given pointer.
    ///
    /// This functions is equivalent to [`NonNull::drop_in_place`].
    ///
    /// # Safety
    ///
    /// The pointer must be [safe to drop], in the context of the inherited
    /// pointer-like equivalence.
    ///
    /// [safe to drop]: https://doc.rust-lang.org/nightly/core/ptr/fn.drop_in_place.html#safety
    unsafe fn drop_in_place(to_drop: Self::Ptr);
}

/// A generalized pointer metadata type with tracing information.
///
/// Unlike the [`Collect<'gc>`] trait, which is implemented by the pointee
/// type, this trait is implemented by the pointer metadata type to support
/// collecting on generalized pointer types.
///
/// # Safety
///
/// The implementation must be [valid], in the context of the inherited
/// pointer-like equivalence.
///
/// [valid]: crate::collect::Collect#safety
pub unsafe trait MetaCollect<'gc, 'a, T: ?Sized + 'a>: MetaLayout<'a, T> {
    /// Returns whether the pointee type needs to be traced.
    ///
    /// Unlike [`Collect::NEEDS_TRACE`], this check is provided by a function
    /// to support collecting on generalized pointers with dynamic metadatas.
    fn needs_trace(self) -> bool;

    /// Traces the pointee type via a generalized reference.
    fn trace<C: Trace<'gc>>(t: Self::Ref, cc: &mut C);
}

// Implementation for sized types.

macro_rules! impl_sized {
    ($($t:ty $(: ($($bounds:tt)*))?),* $(,)?) => {$(
        unsafe impl<'a, T: 'a, $($($bounds)*)?> PtrMetadata<'a, T, ()> for $t {
            fn ptr_metadata(self) {}
        }

        unsafe impl<'a, T: 'a, $($($bounds)*)?> Metadata<'a, T, ()> for $t {
            impl_ptr_metadata!(T);
        }

        unsafe impl<'a, T: 'a, $($($bounds)*)?> MetaLayout<'a, T, ()> for $t {
            #[inline]
            fn layout(self) -> Result<Layout, LayoutError> {
                Ok(Layout::new::<T>())
            }

            #[inline]
            unsafe fn drop_in_place(to_drop: NonNull<T>) {
                unsafe { to_drop.drop_in_place() };
            }
        }

        unsafe impl<'gc, 'a, T, $($($bounds)*)?> MetaCollect<'gc, 'a, T> for $t
        where
            T: Collect<'gc> + 'a,
        {
            #[inline]
            fn needs_trace(self) -> bool {
                T::NEEDS_TRACE
            }

            #[inline]
            fn trace<C: Trace<'gc>>(t: &'a T, cc: &mut C) {
                t.trace(cc);
            }
        }
    )*};
}

impl_sized! {
    (),
    usize,
    DynMetadata<Dyn>: (Dyn: ?Sized),
}

// Implementation for slices.

unsafe impl<'a, T: 'a> PtrMetadata<'a, [T], usize> for usize {
    fn ptr_metadata(self) -> usize {
        self
    }
}

unsafe impl<'a, T: 'a> Metadata<'a, [T]> for usize {
    impl_ptr_metadata!([T]);
}

unsafe impl<'a, T: 'a> MetaLayout<'a, [T]> for usize {
    #[inline]
    fn layout(self) -> Result<Layout, LayoutError> {
        Layout::array::<T>(self)
    }

    #[inline]
    unsafe fn drop_in_place(to_drop: NonNull<[T]>) {
        unsafe { to_drop.drop_in_place() };
    }
}

unsafe impl<'gc, 'a, T> MetaCollect<'gc, 'a, [T]> for usize
where
    T: Collect<'gc> + 'a,
{
    #[inline]
    fn needs_trace(self) -> bool {
        <[T]>::NEEDS_TRACE
    }

    #[inline]
    fn trace<C: Trace<'gc>>(t: &'a [T], cc: &mut C) {
        t.trace(cc);
    }
}

// Implementation for strings.

unsafe impl<'a> PtrMetadata<'a, str, usize> for usize {
    fn ptr_metadata(self) -> usize {
        self
    }
}

unsafe impl<'a> Metadata<'a, str> for usize {
    impl_ptr_metadata!(str);
}

unsafe impl<'a> MetaLayout<'a, str> for usize {
    #[inline]
    fn layout(self) -> Result<Layout, LayoutError> {
        Layout::array::<u8>(self)
    }

    #[inline]
    unsafe fn drop_in_place(_: NonNull<str>) {
        // Strings are just arrays of bytes, which doesn't need drop.
    }
}

unsafe impl<'gc, 'a> MetaCollect<'gc, 'a, str> for usize {
    #[inline]
    fn needs_trace(self) -> bool {
        const _: () = assert!(!str::NEEDS_TRACE);
        false
    }

    #[inline]
    fn trace<C: Trace<'gc>>(_: &'a str, _: &mut C) {
        // Strings are just arrays of bytes, which doesn't need trace.
    }
}

// Implementation for trait objects.

unsafe impl<'a, U, Dyn> PtrMetadata<'a, U, DynMetadata<Dyn>> for DynMetadata<Dyn>
where
    U: ?Sized + Pointee<Metadata = Self> + 'a,
    Dyn: ?Sized,
{
    fn ptr_metadata(self) -> DynMetadata<Dyn> {
        self
    }
}

unsafe impl<'a, U, Dyn> Metadata<'a, U, DynMetadata<Dyn>> for DynMetadata<Dyn>
where
    U: ?Sized + Pointee<Metadata = Self> + 'a,
    Dyn: ?Sized,
{
    impl_ptr_metadata!(U);
}

// Implementation for custom-layout bytes.

unsafe impl<'a> PtrMetadata<'a, [u8], usize> for Layout {
    fn ptr_metadata(self) -> usize {
        self.size()
    }
}

unsafe impl<'a> Metadata<'a, [u8]> for Layout {
    impl_ptr_metadata!([u8]);
}

unsafe impl<'a> MetaLayout<'a, [u8]> for Layout {
    fn layout(self) -> Result<Layout, LayoutError> {
        Ok(self)
    }

    unsafe fn drop_in_place(_: Self::Ptr) {
        // Bytes are just arrays of bytes, which doesn't need drop.
    }
}

unsafe impl<'gc, 'a> MetaCollect<'gc, 'a, [u8]> for Layout {
    #[inline]
    fn needs_trace(self) -> bool {
        const _: () = assert!(!<[u8]>::NEEDS_TRACE);
        false
    }

    #[inline]
    fn trace<C: Trace<'gc>>(_: &'a [u8], _: &mut C) {
        // Bytes are just arrays of bytes, which doesn't need trace.
    }
}

unsafe impl<'a> PtrMetadata<'a, [MaybeUninit<u8>], usize> for Layout {
    fn ptr_metadata(self) -> usize {
        self.size()
    }
}

unsafe impl<'a> Metadata<'a, [MaybeUninit<u8>]> for Layout {
    impl_ptr_metadata!([MaybeUninit<u8>]);
}

unsafe impl<'a> MetaLayout<'a, [MaybeUninit<u8>]> for Layout {
    fn layout(self) -> Result<Layout, LayoutError> {
        Ok(self)
    }

    unsafe fn drop_in_place(_: Self::Ptr) {
        // Bytes are just arrays of bytes, which doesn't need drop.
    }
}

unsafe impl<'gc, 'a> MetaCollect<'gc, 'a, [MaybeUninit<u8>]> for Layout {
    #[inline]
    fn needs_trace(self) -> bool {
        const _: () = assert!(!<[MaybeUninit<u8>]>::NEEDS_TRACE);
        false
    }

    #[inline]
    fn trace<C: Trace<'gc>>(_: &'a [MaybeUninit<u8>], _: &mut C) {
        // Bytes are just arrays of bytes, which doesn't need trace.
    }
}
