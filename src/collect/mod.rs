use crate::gc::{Gc, Unique, Weak};

mod imp;
mod static_;

pub use gc_arena_derive::Collect;
pub use static_::Static;

/// A trait for garbage collected objects that can be placed into `Gc` pointers.
/// This trait is unsafe, because `Gc` pointers inside an Arena are assumed
/// never to be dangling, and in order to ensure this certain rules must be
/// followed (see the safety section).
///
/// It is, however, possible to implement this trait safely by procedurally
/// deriving it (see [`gc_arena_derive::Collect`]), which requires that every
/// field in the structure also implement `Collect`, and ensures that `Drop`
/// cannot safely be implemented. Internally mutable types like `Cell` and
/// `RefCell` do not implement `Collect` in such a way that it is possible to
/// store `Gc` pointers inside them, so write barrier requirements cannot be
/// broken when procedurally deriving `Collect`. A safe way of providing
/// internal mutability in this case is to use [`crate::lock::Lock<T>`] and
/// [`crate::lock::RefLock<T>`], which provides internal mutability
/// while ensuring that write barriers are correctly executed.
///
/// # Safety
///
///   1. `Collect::trace` *must* trace over *every* `Gc` and `Weak` pointer held
///      inside this type.
///   2. Held `Gc` and `Weak` pointers must not be accessed inside `Drop::drop`
///      since during drop any such pointer may be dangling.
///   3. Internal mutability *must* not be used to adopt new `Gc` or `Weak`
///      pointers without calling appropriate write barrier operations during
///      the same arena mutation.
///   4. Structures consisting of `&'gc T` or `&'gc mut T` must not be
///      `Collect<'gc>` unless their backing GC pointers are collected at the
///      same call site.
pub unsafe trait Collect<'gc> {
    /// As an optimization, if this type can never hold a `Gc` pointer and
    /// `trace` is unnecessary to call, you may set this to `false`. The
    /// default value is `true`, signaling that `Collect::trace` must be
    /// called.
    const NEEDS_TRACE: bool = true;

    /// *Must* call [`Trace::trace_gc`] (resp. [`Trace::trace_weak`]) on all
    /// directly owned [`Gc`] (resp. [`Weak`]) pointers. If this type holds
    /// inner types that implement `Collect`, a valid implementation would
    /// simply call [`Trace::trace`] on all the held values to ensure this.
    ///
    /// # Tracing pointers
    ///
    /// [`Gc`] and [`Weak`] have their own implementations of `Collect` which in
    /// turn call [`Trace::trace_gc`] and [`Trace::trace_weak`] respectively.
    /// Because of this, it is not actually ever necessary to call
    /// [`Trace::trace_gc`] and [`Trace::trace_weak`] directly, but be careful!
    /// It is important that owned pointers *themselves* are traced and NOT
    /// their contents (the content type will usually also implement `Collect`,
    /// so this is easy to accidentally do).
    ///
    /// It is always okay to use the [`Trace::trace_gc`] and
    /// [`Trace::trace_weak`] directly as a potentially less risky alternative
    /// when manually implementing `Collect`.
    #[inline]
    #[allow(unused_variables)]
    fn trace<T: Trace<'gc>>(&self, cc: &mut T) {}
}

/// The trait that is passed to the [`Collect::trace`] method.
///
/// Though [`Collect::trace`] is primarily used during the marking phase of the
/// mark and sweep collector, this is a trait rather than a concrete type to
/// facilitate other kinds of uses.
///
/// This trait is not itself unsafe, but implementers of [`Collect`] *must*
/// uphold the safety guarantees of [`Collect`] when using this trait.
pub trait Trace<'gc> {
    /// Trace a [`Gc`] pointer (of any real type).
    fn trace_gc<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: Gc<'gc, T, M>);

    /// Trace a [`Weak`] pointer (of any real type).
    fn trace_weak<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: Weak<'gc, T, M>);

    /// Trace a [`Unique`] pointer (of any real type).
    fn trace_unique<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: &Unique<'gc, T, M>);

    /// This is a convenience method that calls [`Collect::trace`] but
    /// automatically adds a [`Collect::NEEDS_TRACE`] check around it.
    ///
    /// Manual implementations of [`Collect`] are encouraged to use this to
    /// ensure that there is a [`Collect::NEEDS_TRACE`] check without having to
    /// implement it manually. This can be important in cases where the
    /// [`Collect::trace`] method impl is not `#[inline]` or does not have its
    /// own early exit.
    ///
    /// There is generally no need for custom `Trace` implementations to
    /// override this method.
    #[inline]
    fn trace<C: Collect<'gc> + ?Sized>(&mut self, value: &C)
    where
        Self: Sized,
    {
        if C::NEEDS_TRACE {
            value.trace(self);
        }
    }
}
