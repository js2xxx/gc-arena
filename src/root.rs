use core::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem, ptr,
};

use crate::{Collect, Gc, Mutation, Rootable, arena::Root, types::Invariant, vec::Vec};

/// A dynamic root set.
///
/// Users can enable dynamic rooting of values by using this type. Unlike
/// general static rootable types, the storage is resided outside of the arena.
/// This allows for [GC root handles] to be stored outside the arena itself with
/// a lifetime still bound to this root set.
///
/// This is also a special [`Rootable`] type by reference, since the general
/// rootable shim provided by [`macro@Rootable`] is `'static`. This also means
/// this type cannot be stored as a field in other static root types. The reason
/// for the reference style is that the root set's lifetime cannot be bound
/// within the arena itself because of the limits of the Rust type system.
///
/// [GC root handles]: Rooted
pub struct RootSet(Cell<Option<Gc<'static, InnerCell<'static>>>>);

/// A view of [`RootSet`] inside the arena.
///
/// This type cannot be created from its own. Instead, it is created from
/// an associated `RootSet`. It can be used to [stash] and [fetch] rooted types
/// via [root handles].
///
/// [stash]: RootedSet::stash
/// [fetch]: RootedSet::fetch
/// [root handles]: Rooted
#[repr(transparent)]
pub struct RootedSet<'root_set, 'gc> {
    root_set: &'root_set RootSet,
    _marker: (Invariant<'root_set>, Invariant<'gc>),
}

struct InnerCell<'gc>(RefCell<Inner<'gc>>);

const FREE_UNAVAILABLE: usize = usize::MAX;

struct Inner<'gc> {
    slots: Vec<'gc, Slot<'gc>>,
    free_next: usize,
}

enum Slot<'gc> {
    Occupied { root: Gc<'gc, ()>, ref_count: usize },
    Vacant(usize),
}

/// A root handle to a GC'd pointer in a [`RootSet`].
///
/// This type is created by [`RootedSet::stash`] and can be used to [fetch] the
/// GC'd pointer back out of the root set.
///
/// The stashed root is reference-counted and will be automatically dropped
/// when all the handles are dropped.
///
/// [fetch]: RootedSet::fetch
pub struct Rooted<'root_set, R: Rootable> {
    index: usize,
    root_set: &'root_set RootSet,
    marker: Invariant<'root_set, R>,
}

impl crate::sealed::Sealed for &mut RootSet {}
impl<'root_set> Rootable for &'root_set mut RootSet {
    type Root<'gc> = RootedSet<'root_set, 'gc>;
}

impl<'root_set, 'gc> Drop for RootedSet<'root_set, 'gc> {
    fn drop(&mut self) {
        self.root_set.0.set(None);
    }
}

unsafe impl<'gc> Collect<'gc> for RootedSet<'_, 'gc> {
    fn trace<T: crate::collect::Trace<'gc>>(&mut self, cc: &mut T) {
        // Prevents the inner to be traced twice.
        if let Some(mut inner) = unsafe { self.take_inner() } {
            cc.trace_gc(&mut inner);
            unsafe { self.set_inner(inner) };
        }
    }
}

unsafe impl<'gc> Collect<'gc> for InnerCell<'gc> {
    fn trace<T: crate::collect::Trace<'gc>>(&mut self, cc: &mut T) {
        self.0.borrow_mut().slots.trace(cc);
    }
}

unsafe impl<'gc> Collect<'gc> for Slot<'gc> {
    fn trace<T: crate::collect::Trace<'gc>>(&mut self, cc: &mut T) {
        match self {
            Slot::Occupied { root, .. } => root.trace(cc),
            Slot::Vacant(_) => {}
        }
    }
}

impl RootSet {
    /// Creates a new empty root set.
    ///
    /// This type is unusable on its own. See [`RootedSet::new`] for more
    /// information.
    pub const fn new() -> Self {
        Self(Cell::new(None))
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

impl<'root_set, 'gc> RootedSet<'root_set, 'gc> {
    /// Creates a new root set view inside an `Arena`.
    ///
    /// A shorthand for direct construction inside an `Arena` is also available
    /// via [`Arena::with_root_set`].
    ///
    /// [`Arena::with_root_set`]: crate::Arena::with_root_set
    pub fn new(root_set: &'root_set mut RootSet, mc: &Mutation<'gc>) -> Self {
        let set = RootedSet {
            root_set,
            _marker: (PhantomData, PhantomData),
        };
        assert!(set.root_set.0.get().is_none());

        let t = RefCell::new(Inner {
            slots: Vec::new(mc),
            free_next: FREE_UNAVAILABLE,
        });
        unsafe { set.set_inner(Gc::new(mc, InnerCell(t))) };

        set
    }

    unsafe fn inner(&self) -> Gc<'gc, InnerCell<'gc>> {
        unsafe {
            mem::transmute::<Gc<'static, InnerCell<'static>>, Gc<'gc, InnerCell<'gc>>>(
                self.root_set.0.get().expect("uninitialized root set"),
            )
        }
    }

    unsafe fn take_inner(&self) -> Option<Gc<'gc, InnerCell<'gc>>> {
        Some(unsafe {
            mem::transmute::<Gc<'static, InnerCell<'static>>, Gc<'gc, InnerCell<'gc>>>(
                self.root_set.0.take()?,
            )
        })
    }

    unsafe fn set_inner(&self, inner: Gc<'gc, InnerCell<'gc>>) {
        self.root_set.0.set(Some(unsafe {
            mem::transmute::<Gc<'gc, InnerCell<'gc>>, Gc<'static, InnerCell<'static>>>(inner)
        }));
    }

    /// Tries to fetch a value from the root set.
    ///
    /// # Returns
    ///
    /// `None` if the root handle does not belong to this root set.
    pub fn try_fetch<R: Rootable>(
        &self,
        r: &Rooted<'root_set, R>,
    ) -> Option<Gc<'gc, Root<'gc, R>>> {
        if !ptr::eq(r.root_set, self.root_set) {
            return None;
        }

        let inner = unsafe { self.inner() };
        let inner = inner.0.borrow();
        match inner.slots[r.index] {
            Slot::Occupied { root, .. } => Some(unsafe { Gc::cast(root) }),
            Slot::Vacant(_) => unreachable!("rooted slot is vacant"),
        }
    }

    /// Fetches a value from the root set.
    ///
    /// # Panics
    ///
    /// Panics if the root handle does not belong to this root set.
    pub fn fetch<R: Rootable>(&self, r: &Rooted<'root_set, R>) -> Gc<'gc, Root<'gc, R>> {
        self.try_fetch(r)
            .expect("rooted value does not belong to this root set")
    }

    /// Stashes a value into the root set.
    pub fn stash<R>(&self, mc: &Mutation<'gc>, root: Gc<'gc, Root<'gc, R>>) -> Rooted<'root_set, R>
    where
        R: Rootable,
    {
        let inner = unsafe { self.inner() };
        mc.backward_barrier(inner, Some(root));

        let value = Slot::Occupied {
            root: unsafe { Gc::cast(root) },
            ref_count: 1,
        };
        let inner = &mut *inner.0.borrow_mut();

        let free_next = inner.free_next;
        let index = match inner.slots.get_mut(free_next) {
            None => {
                let new = inner.slots.len();
                inner.slots.push(mc, value);
                new
            }
            Some(slot) => {
                inner.free_next = match slot {
                    Slot::Occupied { .. } => unreachable!("free slot is occupied"),
                    Slot::Vacant(next) => *next,
                };
                *slot = value;
                free_next
            }
        };

        Rooted {
            index,
            root_set: self.root_set,
            marker: PhantomData,
        }
    }
}

impl<R: Rootable> Clone for Rooted<'_, R> {
    fn clone(&self) -> Self {
        let cell = self.root_set.0.get().unwrap();
        let mut inner = cell.0.borrow_mut();

        match &mut inner.slots[self.index] {
            Slot::Occupied { ref_count, .. } => *ref_count += 1,
            Slot::Vacant(_) => unreachable!("rooted slot is vacant"),
        }

        Self {
            index: self.index,
            root_set: self.root_set,
            marker: PhantomData,
        }
    }
}

impl<R: Rootable> Drop for Rooted<'_, R> {
    fn drop(&mut self) {
        let cell = self.root_set.0.get().unwrap();
        let mut inner = cell.0.borrow_mut();

        let ref_count = match &mut inner.slots[self.index] {
            Slot::Occupied { ref_count, .. } => ref_count,
            Slot::Vacant(_) => unreachable!("rooted slot is vacant"),
        };

        if *ref_count == 1 {
            inner.slots[self.index] = Slot::Vacant(inner.free_next);
            inner.free_next = self.index;
        } else {
            *ref_count -= 1;
        }
    }
}
