use core::{
    alloc::Layout,
    borrow::Borrow,
    convert::Infallible,
    fmt::{self, Debug, Display, Pointer},
    hash::{Hash, Hasher},
    marker::{PhantomData, Unsize},
    mem::MaybeUninit,
    ops::Deref,
    ptr::{NonNull, Pointee},
};

use crate::{
    Finalization,
    barrier::{Unlock, Write},
    collect::{Collect, Static, Trace},
    context::Mutation,
    types::{GcBox, GcBoxHeader, GcBoxInner, GcColor, Invariant, MetaLayout},
};

mod unique;
mod weak;

pub use self::{unique::Unique, weak::Weak};

/// A garbage collected pointer to a type T. Implements Copy, and is implemented as a plain machine
/// pointer. You can only allocate `Gc` pointers through a `&Mutation<'gc>` inside an arena type,
/// and through "generativity" such `Gc` pointers may not escape the arena they were born in or
/// be stored inside TLS. This, combined with correct `Collect` implementations, means that `Gc`
/// pointers will never be dangling and are always safe to access.
#[repr(transparent)]
pub struct Gc<'gc, T: ?Sized + 'gc> {
    pub(crate) ptr: GcBox,
    pub(crate) _invariant: Invariant<'gc, T>,
}

impl<'gc, T: Debug + ?Sized + 'gc> Debug for Gc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc> Pointer for Gc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Pointer::fmt(&Gc::as_ptr(*self), fmt)
    }
}

impl<'gc, T: Display + ?Sized + 'gc> Display for Gc<'gc, T> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(&**self, fmt)
    }
}

impl<'gc, T: ?Sized + 'gc> Copy for Gc<'gc, T> {}

impl<'gc, T: ?Sized + 'gc> Clone for Gc<'gc, T> {
    #[inline]
    fn clone(&self) -> Gc<'gc, T> {
        *self
    }
}

unsafe impl<'gc, T: ?Sized + 'gc> Collect<'gc> for Gc<'gc, T> {
    #[inline]
    fn trace<C: Trace<'gc>>(&self, cc: &mut C) {
        cc.trace_gc(*self)
    }
}

impl<'gc, T: ?Sized + 'gc> Deref for Gc<'gc, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        unsafe { &*self.ptr.unerased_value::<T>() }
    }
}

impl<'gc, T: ?Sized + 'gc> AsRef<T> for Gc<'gc, T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<'gc, T: ?Sized + 'gc> Borrow<T> for Gc<'gc, T> {
    #[inline]
    fn borrow(&self) -> &T {
        self
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

    pub fn new_unsize<Dyn>(mc: &Mutation<'gc>, t: T) -> Gc<'gc, Dyn>
    where
        T: Collect<'gc> + Unsize<Dyn>,
        Dyn: ?Sized + 'gc,
        <Dyn as Pointee>::Metadata: MetaLayout<Dyn>,
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
}

impl<'gc, T: 'static> Gc<'gc, T> {
    /// Create a new `Gc` pointer from a static value.
    ///
    /// This method does not require that the type `T` implement `Collect`. This uses [`Static`]
    /// internally to automatically provide a trivial `Collect` impl and is equivalent to the
    /// following code:
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
        unsafe { Gc::cast::<T>(p) }
    }
}

impl<'gc, T: ?Sized + 'gc> Gc<'gc, T> {
    /// Cast a `Gc` pointer to a different type.
    ///
    /// # Safety
    /// It must be valid to dereference a `*mut U` that has come from casting a `*mut T`.
    #[inline]
    pub unsafe fn cast<U: 'gc>(this: Gc<'gc, T>) -> Gc<'gc, U> {
        Gc {
            ptr: this.ptr,
            _invariant: PhantomData,
        }
    }

    /// Retrieve a `Gc` from a raw pointer obtained from `Gc::as_ptr`
    ///
    /// # Safety
    /// The provided pointer must have been obtained from `Gc::as_ptr`, and the pointer must not
    /// have been collected yet.
    #[inline]
    pub unsafe fn from_ptr(ptr: *const T) -> Gc<'gc, T> {
        let layout = Layout::new::<GcBoxHeader>();
        // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
        let (_, header_offset) = layout
            .extend(unsafe { Layout::for_value_raw(ptr) })
            .unwrap();
        // SAFETY: `ptr` is previously obtained from `Gc::as_ptr`, so there is always a header.
        let ptr = unsafe { (ptr as *mut T).byte_sub(header_offset) } as *mut GcBoxInner<T>;
        Gc {
            // SAFETY: `ptr` is valid and aligned guaranteed by the caller.
            ptr: unsafe { GcBox::erase(NonNull::new_unchecked(ptr)) },
            _invariant: PhantomData,
        }
    }
}

impl<'gc, T: Unlock + ?Sized + 'gc> Gc<'gc, T> {
    /// Shorthand for [`Gc::write`]`(mc, self).`[`unlock()`](Write::unlock).
    #[inline]
    pub fn unlock(self, mc: &Mutation<'gc>) -> &'gc T::Unlocked {
        Gc::write(mc, self);
        // SAFETY: see doc-comment.
        unsafe { self.get_ref().unlock_unchecked() }
    }
}

impl<'gc, T: ?Sized + 'gc> Gc<'gc, T> {
    /// Obtains a long-lived reference to the contents of this `Gc`.
    ///
    /// Unlike `AsRef` or `Deref`, the returned reference isn't bound to the `Gc` itself, and
    /// will stay valid for the entirety of the current arena callback.
    #[inline]
    pub fn get_ref(self) -> &'gc T {
        // SAFETY: The returned reference cannot escape the current arena callback, as `&'gc T`
        // never implements `Collect` (unless `'gc` is `'static`, which is impossible here), and
        // so cannot be stored inside the GC root.
        unsafe { &*self.ptr.unerased_value::<T>() }
    }

    #[inline]
    pub fn downgrade(this: Gc<'gc, T>) -> Weak<'gc, T> {
        Weak { inner: this }
    }

    /// Triggers a write barrier on this `Gc`, allowing for safe mutation.
    ///
    /// This triggers an unrestricted *backwards* write barrier on this pointer, meaning that it is
    /// guaranteed that this pointer can safely adopt *any* arbitrary child pointers (until the next
    /// time that collection is triggered).
    ///
    /// It returns a reference to the inner `T` wrapped in a `Write` marker to allow for
    /// unrestricted mutation on the held type or any of its directly held fields.
    #[inline]
    pub fn write(mc: &Mutation<'gc>, gc: Self) -> &'gc Write<T> {
        unsafe {
            mc.backward_barrier::<_, Infallible>(gc, None);
            // SAFETY: the write barrier stays valid until the end of the current callback.
            Write::assume(gc.get_ref())
        }
    }

    /// Returns true if two `Gc`s point to the same allocation.
    ///
    /// Similarly to `Rc::ptr_eq` and `Arc::ptr_eq`, this function ignores the metadata of `dyn`
    /// pointers.
    #[inline]
    pub fn ptr_eq(this: Gc<'gc, T>, other: Gc<'gc, T>) -> bool {
        // TODO: Equivalent to `core::ptr::addr_eq`:
        // https://github.com/rust-lang/rust/issues/116324
        Gc::as_ptr(this) as *const () == Gc::as_ptr(other) as *const ()
    }

    #[inline]
    pub fn as_ptr(gc: Gc<'gc, T>) -> *const T {
        unsafe { gc.ptr.unerased_value::<T>() }
    }

    /// Returns true when a pointer is *dead* during finalization. This is equivalent to
    /// `Weak::is_dead` for strong pointers.
    ///
    /// Any strong pointer reachable from the root will never be dead, BUT there can be strong
    /// pointers reachable only through other weak pointers that can be dead.
    #[inline]
    pub fn is_dead(_: &Finalization<'gc>, gc: Gc<'gc, T>) -> bool {
        matches!(gc.ptr.header().color(), GcColor::White | GcColor::WhiteWeak)
    }

    /// Manually marks a dead `Gc` pointer as reachable and keeps it alive.
    ///
    /// Equivalent to `Weak::resurrect` for strong pointers. Manually marks this pointer and
    /// all transitively held pointers as reachable, thus keeping them from being dropped this
    /// collection cycle.
    #[inline]
    pub fn resurrect(fc: &Finalization<'gc>, gc: Gc<'gc, T>) {
        fc.resurrect(gc.ptr);
    }
}

impl<'gc, T: PartialEq + ?Sized + 'gc> PartialEq for Gc<'gc, T> {
    fn eq(&self, other: &Self) -> bool {
        (**self).eq(other)
    }
}

impl<'gc, T: Eq + ?Sized + 'gc> Eq for Gc<'gc, T> {}

impl<'gc, T: PartialOrd + ?Sized + 'gc> PartialOrd for Gc<'gc, T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        (**self).partial_cmp(other)
    }

    fn le(&self, other: &Self) -> bool {
        (**self).le(other)
    }

    fn lt(&self, other: &Self) -> bool {
        (**self).lt(other)
    }

    fn ge(&self, other: &Self) -> bool {
        (**self).ge(other)
    }

    fn gt(&self, other: &Self) -> bool {
        (**self).gt(other)
    }
}

impl<'gc, T: Ord + ?Sized + 'gc> Ord for Gc<'gc, T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (**self).cmp(other)
    }
}

impl<'gc, T: Hash + ?Sized + 'gc> Hash for Gc<'gc, T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}
