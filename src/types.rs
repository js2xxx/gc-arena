use core::alloc::{Layout, LayoutError};
use core::cell::Cell;
use core::marker::{PhantomData, Unsize};
use core::ptr::{DynMetadata, NonNull, Pointee};
use core::{fmt, mem, ptr};

use crate::{collect::Collect, context::Context};

pub trait MetaLayout<T: Pointee + ?Sized>: Copy {
    fn layout(self) -> Result<Layout, LayoutError>;
}

impl<T> MetaLayout<T> for () {
    fn layout(self) -> Result<Layout, LayoutError> {
        Ok(Layout::new::<T>())
    }
}

impl<T> MetaLayout<[T]> for usize {
    fn layout(self) -> Result<Layout, LayoutError> {
        Layout::array::<T>(self)
    }
}

impl<Dyn: ?Sized, T: ?Sized + Pointee<Metadata = Self>> MetaLayout<T> for DynMetadata<Dyn> {
    fn layout(self) -> Result<Layout, LayoutError> {
        let ptr: *const T = ptr::from_raw_parts(ptr::null::<()>(), self);
        // SAFETY: The metadata part is valid.
        Ok(unsafe { Layout::for_value_raw(ptr) })
    }
}

/// A thin-pointer-sized box containing a type-erased GC object.
/// Stores the metadata required by the GC algorithm inline (see `GcBoxInner`
/// for its typed counterpart).

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct GcBox(NonNull<GcBoxInner<()>>);

impl GcBox {
    /// # Safety
    ///
    /// `ptr` must point to a valid `GcBoxInner`.
    pub(crate) unsafe fn from_raw(ptr: NonNull<GcBoxInner<()>>) -> Self {
        Self(ptr)
    }

    /// Erases a pointer to a typed GC object.
    ///
    /// **SAFETY:** The pointer must point to a valid `GcBoxInner` allocated
    /// in a `Box`.
    #[inline(always)]
    pub(crate) unsafe fn erase<T: ?Sized>(ptr: NonNull<GcBoxInner<T>>) -> Self {
        // This cast is sound because `GcBoxInner` is `repr(C)`.
        unsafe {
            let (erased, metadata) = ptr.to_raw_parts();
            let gc_box = Self(erased.cast());
            debug_assert_eq!(gc_box.metadata::<T>(), metadata);
            gc_box
        }
    }

    /// Gets a pointer to the value stored inside this box.
    /// `T` must be the same type that was used with `erase`, so that
    /// we can correctly compute the field offset.
    #[inline(always)]
    pub(crate) unsafe fn unerased_value<T: ?Sized>(&self) -> *mut T {
        unsafe {
            let metadata = self.metadata::<T>();
            let ptr: *mut GcBoxInner<T> = ptr::from_raw_parts_mut(self.0.as_ptr(), metadata);
            // Don't create a reference, to keep the full provenance.
            // Also, this gives us interior mutability "for free".
            (&raw mut (*ptr).value) as *mut T
        }
    }

    unsafe fn metadata<T: ?Sized>(&self) -> <T as Pointee>::Metadata {
        let offset = GcBoxInner::<T>::METADATA_OFFSET;
        unsafe {
            let ptr = self.0.byte_sub(offset);
            ptr.cast().read()
        }
    }

    #[inline(always)]
    pub(crate) fn header(&self) -> &GcBoxHeader {
        unsafe { &self.0.as_ref().header }
    }

    /// Returns the (shallow) size occupied by this box in memory.
    #[inline(always)]
    pub(crate) fn size_of_box(&self) -> usize {
        let (layout, _) = unsafe { (self.header().vtable().box_layout)(*self) };
        layout.size()
    }

    /// Traces the stored value.
    ///
    /// **SAFETY**: `Self::drop_in_place` must not have been called.
    #[inline(always)]
    pub(crate) unsafe fn trace_value(&self, cc: &mut Context) {
        unsafe { (self.header().vtable().trace_value)(*self, cc) }
    }

    /// Drops the stored value.
    ///
    /// **SAFETY**: once called, no GC pointers should access the stored value
    /// (but accessing the `GcBox` itself is still safe).
    #[inline(always)]
    pub(crate) unsafe fn drop_in_place(&mut self) {
        unsafe { (self.header().vtable().drop_value)(*self) }
    }

    /// Deallocates the box. Failing to call `Self::drop_in_place` beforehand
    /// will cause the stored value to be leaked.
    ///
    /// **SAFETY**: once called, this `GcBox` should never be accessed by any GC
    /// pointers again.
    #[inline(always)]
    pub(crate) unsafe fn dealloc(self) {
        unsafe {
            let (layout, offset) = (self.header().vtable().box_layout)(self);
            let ptr = self.0.as_ptr().byte_sub(offset) as *mut u8;
            // SAFETY: the pointer was `Box`-allocated with this layout.
            alloc::alloc::dealloc(ptr, layout);
        }
    }
}

pub(crate) struct GcBoxHeader {
    /// The next element in the global linked list of allocated objects.
    next: Cell<Option<GcBox>>,
    /// The next element in the gray list.
    gray_next: Cell<Option<GcBox>>,
    /// A custom virtual function table for handling type-specific operations.
    ///
    /// The lower bits of the pointer are used to store GC flags:
    /// - bits 0 & 1 for the current `GcColor`;
    /// - bit 2 for the `needs_trace` flag;
    /// - bit 3 for the `is_live` flag.
    tagged_vtable: Cell<*const CollectVTable>,
}

// Helper trait to materialize vtables in static memory.
trait HasCollectVTable {
    const VTABLE: CollectVTable;
}

impl<'gc, T> HasCollectVTable for T
where
    T: Collect<'gc> + ?Sized,
    <T as Pointee>::Metadata: MetaLayout<T>,
{
    const VTABLE: CollectVTable = CollectVTable::new::<T>();
}

// Helper trait to materialize vtables in static memory.
trait HasCollectVTableUnsize<U: ?Sized> {
    const VTABLE: CollectVTable;
}

impl<'gc, T, U> HasCollectVTableUnsize<U> for T
where
    T: Collect<'gc> + Unsize<U> + ?Sized,
    U: ?Sized,
    <U as Pointee>::Metadata: MetaLayout<U>,
{
    const VTABLE: CollectVTable = CollectVTable::new_unsize::<T, U>();
}

impl GcBoxHeader {
    pub const fn new<'gc, T>() -> Self
    where
        T: Collect<'gc> + ?Sized,
        <T as Pointee>::Metadata: MetaLayout<T>,
    {
        let vtable: &'static _ = &<T as HasCollectVTable>::VTABLE;
        Self {
            next: Cell::new(None),
            gray_next: Cell::new(None),
            tagged_vtable: Cell::new(ptr::from_ref(vtable)),
        }
    }
    pub const fn new_unsize<'gc, T, U>() -> Self
    where
        T: Collect<'gc> + Unsize<U> + ?Sized,
        U: ?Sized,
        <U as Pointee>::Metadata: MetaLayout<U>,
    {
        let vtable: &'static _ = &<T as HasCollectVTableUnsize<U>>::VTABLE;
        Self {
            next: Cell::new(None),
            gray_next: Cell::new(None),
            tagged_vtable: Cell::new(ptr::from_ref(vtable)),
        }
    }

    /// # Safety
    ///
    /// `T` must conform to the type that was used with `new`.
    pub unsafe fn reset_vtable<'gc, T>(&self)
    where
        T: Collect<'gc> + ?Sized,
        <T as Pointee>::Metadata: MetaLayout<T>,
    {
        let vtable: &'static _ = &<T as HasCollectVTable>::VTABLE;
        let tags = tagged_ptr::get::<0xf, _>(self.tagged_vtable.get());
        self.tagged_vtable
            .set(ptr::from_ref(vtable).map_addr(|addr| addr | tags));
        self.set_needs_trace(T::NEEDS_TRACE);
    }

    /// # Safety
    ///
    /// `T` and `U` must conform to the type that was used with `new`.
    #[expect(unused)]
    pub unsafe fn reset_vtable_unsize<'gc, T, U>(&self)
    where
        T: Collect<'gc> + Unsize<U> + ?Sized,
        U: ?Sized,
        <U as Pointee>::Metadata: MetaLayout<U>,
    {
        let vtable: &'static _ = &<T as HasCollectVTableUnsize<U>>::VTABLE;
        let tags = tagged_ptr::get::<0xf, _>(self.tagged_vtable.get());
        self.tagged_vtable
            .set(ptr::from_ref(vtable).map_addr(|addr| addr | tags));
        self.set_needs_trace(T::NEEDS_TRACE);
    }

    /// Gets a reference to the `CollectVTable` used by this box.
    #[inline(always)]
    pub(crate) fn vtable(&self) -> &'static CollectVTable {
        let ptr = tagged_ptr::untag(self.tagged_vtable.get());
        // SAFETY:
        // - the pointer was properly untagged.
        // - the vtable is stored in static memory.
        unsafe { &*ptr }
    }

    /// Gets the next element in the global linked list of allocated objects.
    #[inline(always)]
    pub(crate) fn next(&self) -> Option<GcBox> {
        self.next.get()
    }

    /// Sets the next element in the global linked list of allocated objects.
    #[inline(always)]
    pub(crate) fn set_next(&self, next: Option<GcBox>) {
        self.next.set(next)
    }

    /// Gets the next element in the gray list.
    #[inline(always)]
    pub(crate) fn gray_next(&self) -> Option<GcBox> {
        self.gray_next.get()
    }

    /// Sets the next element in the gray list.
    #[inline(always)]
    pub(crate) fn set_gray_next(&self, next: Option<GcBox>) {
        self.gray_next.set(next)
    }

    #[inline]
    pub(crate) fn color(&self) -> GcColor {
        match tagged_ptr::get::<0x3, _>(self.tagged_vtable.get()) {
            0x0 => GcColor::White,
            0x1 => GcColor::WhiteWeak,
            0x2 => GcColor::Gray,
            _ => GcColor::Black,
        }
    }

    #[inline]
    pub(crate) fn set_color(&self, color: GcColor) {
        tagged_ptr::set::<0x3, _>(
            &self.tagged_vtable,
            match color {
                GcColor::White => 0x0,
                GcColor::WhiteWeak => 0x1,
                GcColor::Gray => 0x2,
                GcColor::Black => 0x3,
            },
        );
    }
    #[inline]
    pub(crate) fn needs_trace(&self) -> bool {
        tagged_ptr::get::<0x4, _>(self.tagged_vtable.get()) != 0x0
    }

    /// Determines whether or not we've dropped the `dyn Collect` value
    /// stored in `GcBox.value`
    /// When we garbage-collect a `GcBox` that still has outstanding weak pointers,
    /// we set `alive` to false. When there are no more weak pointers remaining,
    /// we will deallocate the `GcBox`, but skip dropping the `dyn Collect` value
    /// (since we've already done it).
    #[inline]
    pub(crate) fn is_live(&self) -> bool {
        tagged_ptr::get::<0x8, _>(self.tagged_vtable.get()) != 0x0
    }

    #[inline]
    pub(crate) fn set_needs_trace(&self, needs_trace: bool) {
        tagged_ptr::set_bool::<0x4, _>(&self.tagged_vtable, needs_trace);
    }

    #[inline]
    pub(crate) fn set_live(&self, alive: bool) {
        tagged_ptr::set_bool::<0x8, _>(&self.tagged_vtable, alive);
    }
}

/// Type-specific operations for GC'd values.
///
/// We use a custom vtable instead of `dyn Collect` for extra flexibility.
/// The type is over-aligned so that `GcBoxHeader` can store flags into the LSBs of the vtable pointer.
#[repr(align(16))]
pub(crate) struct CollectVTable {
    box_layout: unsafe fn(GcBox) -> (Layout, usize),
    /// Drops the value stored in the given `GcBox` (without deallocating the box).
    drop_value: unsafe fn(GcBox),
    /// Traces the value stored in the given `GcBox`.
    trace_value: unsafe fn(GcBox, &mut Context),
}

impl PartialEq for CollectVTable {
    fn eq(&self, other: &Self) -> bool {
        // VTable instantiation in miri is not unique per type, so we can't
        // assert the equality of their addresses in miri.
        #[cfg(not(miri))]
        return ptr::eq(self, other);
        #[cfg(miri)]
        return ptr::fn_addr_eq(self.box_layout, other.box_layout)
            && ptr::fn_addr_eq(self.drop_value, other.drop_value)
            && ptr::fn_addr_eq(self.trace_value, other.trace_value);
    }
}

impl fmt::Debug for CollectVTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CollectVTable({self:p})")
    }
}

impl CollectVTable {
    /// Makes a vtable for a known type.
    #[inline(always)]
    const fn new<'gc, T>() -> Self
    where
        T: Collect<'gc> + ?Sized,
        <T as Pointee>::Metadata: MetaLayout<T>,
    {
        Self {
            box_layout: |erased| unsafe {
                GcBoxInner::<T>::box_layout(erased.metadata::<T>()).unwrap_unchecked()
            },
            drop_value: |erased| unsafe {
                ptr::drop_in_place(erased.unerased_value::<T>());
            },
            trace_value: |erased, cc| unsafe {
                let val = &*(erased.unerased_value::<T>());
                val.trace(cc)
            },
        }
    }

    /// Makes a vtable for a known unsize type.
    #[inline(always)]
    const fn new_unsize<'gc, T, U>() -> Self
    where
        T: Collect<'gc> + Unsize<U> + ?Sized,
        U: ?Sized,
        <U as Pointee>::Metadata: MetaLayout<U>,
    {
        Self {
            box_layout: |erased| unsafe {
                GcBoxInner::<U>::box_layout(erased.metadata::<U>()).unwrap_unchecked()
            },
            drop_value: |erased| unsafe {
                ptr::drop_in_place(erased.unerased_value::<T>());
            },
            trace_value: |erased, cc| unsafe {
                let val = &*(erased.unerased_value::<T>());
                val.trace(cc)
            },
        }
    }
}

/// A typed GC'd value, together with its metadata.
/// This type is never manipulated directly by the GC algorithm, allowing
/// user-facing `Gc`s to freely cast their pointer to it.
#[repr(C)]
pub(crate) struct GcBoxInner<T: ?Sized> {
    pub(crate) header: GcBoxHeader,
    /// The typed value stored in this `GcBox`.
    pub(crate) value: mem::ManuallyDrop<T>,
}

impl<T: ?Sized> GcBoxInner<T> {
    pub(crate) const METADATA_OFFSET: usize =
        match Layout::new::<<T as Pointee>::Metadata>().extend(Layout::new::<GcBoxHeader>()) {
            Ok((_, offset)) => offset,
            Err(_) => panic!("Layout calculation failed"),
        };
}

impl<T: ?Sized + Pointee> GcBoxInner<T> {
    pub(crate) fn box_layout(metadata: T::Metadata) -> Result<(Layout, usize), LayoutError>
    where
        T::Metadata: MetaLayout<T>,
    {
        let value = metadata.layout()?;
        let header = Layout::new::<GcBoxHeader>();
        let metadata = Layout::new::<T::Metadata>();

        let (layout, offset) = metadata.extend(header)?;
        let (layout, _) = layout.extend(value)?;
        Ok((layout, offset))
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub(crate) enum GcColor {
    /// An object that has not yet been reached by tracing (if we're in a tracing phase).
    ///
    /// During `Phase::Sweep`, we will free all white objects that existed *before* the start of the
    /// current `Phase::Sweep`. Objects allocated during `Phase::Sweep` will be white, but will not
    /// be freed.
    White,
    /// Like White, but for objects weakly reachable from a Black object.
    ///
    /// These objects may drop their contents during `Phase::Sweep`, but must stay allocated so that
    /// weak references can check the alive status.
    WhiteWeak,
    /// An object reachable from a Black object, but that has not yet been traced using
    /// `Collect::trace`. We also mark black objects as gray during `Phase::Mark` in response to
    /// a write barrier, so that we re-trace and find any objects newly reachable from the mutated
    /// object.
    Gray,
    /// An object that was reached during tracing. It will not be freed during `Phase::Sweep`. At
    /// the end of `Phase::Sweep`, all black objects will be reset to white.
    Black,
}

// Phantom type that holds a lifetime and ensures that it is invariant.
pub(crate) type Invariant<'a, T = ()> = PhantomData<Cell<&'a T>>;

/// Utility functions for tagging and untagging pointers.
mod tagged_ptr {
    use core::cell::Cell;

    trait ValidMask<const MASK: usize> {
        const CHECK: ();
    }

    impl<T, const MASK: usize> ValidMask<MASK> for T {
        const CHECK: () = assert!(MASK < core::mem::align_of::<T>());
    }

    /// Checks that `$mask` can be used to tag a pointer to `$type`.
    /// If this isn't true, this macro will cause a post-monomorphization error.
    macro_rules! check_mask {
        ($type:ty, $mask:expr) => {
            let _ = <$type as ValidMask<$mask>>::CHECK;
        };
    }

    #[inline(always)]
    pub(super) fn untag<T>(tagged_ptr: *const T) -> *const T {
        let mask = core::mem::align_of::<T>() - 1;
        tagged_ptr.map_addr(|addr| addr & !mask)
    }

    #[inline(always)]
    pub(super) fn get<const MASK: usize, T>(tagged_ptr: *const T) -> usize {
        check_mask!(T, MASK);
        tagged_ptr.addr() & MASK
    }

    #[inline(always)]
    pub(super) fn set<const MASK: usize, T>(pcell: &Cell<*const T>, tag: usize) {
        check_mask!(T, MASK);
        let ptr = pcell.get();
        let ptr = ptr.map_addr(|addr| (addr & !MASK) | (tag & MASK));
        pcell.set(ptr)
    }

    #[inline(always)]
    pub(super) fn set_bool<const MASK: usize, T>(pcell: &Cell<*const T>, value: bool) {
        check_mask!(T, MASK);
        let ptr = pcell.get();
        let ptr = ptr.map_addr(|addr| (addr & !MASK) | if value { MASK } else { 0 });
        pcell.set(ptr)
    }
}
