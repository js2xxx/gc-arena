use core::{
    alloc::Allocator,
    cell::Cell,
    cmp::Ordering::Greater,
    marker::PhantomData,
    mem,
    ops::{ControlFlow, Deref, DerefMut},
};

use crate::{
    collect::{Collect, Trace},
    gc::{Gc, Unique, Weak},
    metrics::Metrics,
    ptr::MetaCollect,
    types::{GcBox, GcBoxHeader, GcColor, Invariant},
};

/// Handle value given by arena callbacks during construction and mutation. Allows allocating new
/// `Gc` pointers and internally mutating values held by `Gc` pointers.
pub struct Mutation<'gc> {
    context: &'gc Context,
    alloc: &'gc dyn Allocator,
    _invariant: Invariant<'gc>,
}

impl<'gc> Mutation<'gc> {
    #[inline]
    pub fn metrics(&self) -> &Metrics {
        self.context.metrics()
    }

    /// IF we are in the marking phase AND the `parent` pointer is colored black AND the `child` (if
    /// given) is colored white, then change the `parent` color to gray and enqueue it for tracing.
    ///
    /// This operation is known as a "backwards write barrier". Calling this method is one of the
    /// safe ways for the value in the `parent` pointer to use internal mutability to adopt the
    /// `child` pointer without invalidating the color invariant.
    ///
    /// If the `child` parameter is given, then calling this method ensures that the `parent`
    /// pointer may safely adopt the `child` pointer. If no `child` is given, then calling this
    /// method is more general, and it ensures that the `parent` pointer may adopt *any* child
    /// pointer(s) before collection is next triggered.
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

    /// A version of [`Mutation::backward_barrier`] that allows adopting a [`Weak`] child.
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

    /// IF we are in the marking phase AND the `parent` pointer (if given) is colored black, AND
    /// the `child` is colored white, then immediately change the `child` to gray and enqueue it
    /// for tracing.
    ///
    /// This operation is known as a "forwards write barrier". Calling this method is one of the
    /// safe ways for the value in the `parent` pointer to use internal mutability to adopt the
    /// `child` pointer without invalidating the color invariant.
    ///
    /// If the `parent` parameter is given, then calling this method ensures that the `parent`
    /// pointer may safely adopt the `child` pointer. If no `parent` is given, then calling this
    /// method is more general, and it ensures that the `child` pointer may be adopted by *any*
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

    /// A version of [`Mutation::forward_barrier`] that allows adopting a [`Weak`] child.
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
/// Derefs to `Mutation<'gc>` to allow for arbitrary mutation, but adds additional powers to examine
/// the state of the fully marked arena.
pub struct Finalization<'gc>(Mutation<'gc>);

impl<'gc> Deref for Finalization<'gc> {
    type Target = Mutation<'gc>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'gc> Finalization<'gc> {
    #[inline]
    pub(crate) fn resurrect(&self, gc_box: GcBox) {
        self.context.resurrect(gc_box)
    }
}

impl<'gc> Trace<'gc> for Context {
    fn trace_gc<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: Gc<'gc, T, M>) {
        Context::trace(self, gc.ptr)
    }

    fn trace_weak<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: Weak<'gc, T, M>) {
        Context::trace_weak(self, gc.inner.ptr)
    }

    fn trace_unique<T: ?Sized + 'gc, M: 'gc>(&mut self, gc: &Unique<'gc, T, M>) {
        Context::trace(self, gc.ptr);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum Phase {
    Mark,
    Sweep,
    Sleep,
    Drop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum RunUntil {
    // Run collection until we reach the stop condition *or* debt is zero.
    PayDebt,
    // Run collection until we reach our stop condition.
    Stop,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) enum Stop {
    // Don't proceed past the end of marking, *just* before the sweep phase
    FullyMarked,
    // Don't proceed past the very beginning of the sweep phase
    AtSweep,
    // Stop once we reach the end of the current cycle and are in `Phase::Sleep`.
    FinishCycle,
    // Stop once we have done an entire cycle as a single atomic unit. This is the maximum amount
    // of work that a call to `Context::do_collection` will do, since a full collection as a single
    // atomic unit means that all unreachable values *must* already be freed.
    Full,
}

pub(crate) struct Context {
    metrics: Metrics,
    phase: Phase,
    #[cfg(feature = "tracing")]
    phase_span: tracing::Span,

    // A linked list of all allocated `GcBox`es.
    all: Cell<Option<GcBox>>,

    // A copy of the head of `all` at the end of `Phase::Mark`.
    // During `Phase::Sweep`, we free all white allocations on this list.
    // Any allocations created *during* `Phase::Sweep` will be added to `all`,
    // but `sweep` will *not* be updated. This ensures that we keep allocations
    // alive until we've had a chance to trace them.
    sweep: Option<GcBox>,

    // The most recent black object that we encountered during `Phase::Sweep`.
    // When we free objects, we update this `GcBox.next` to remove them from
    // the linked list.
    sweep_prev: Cell<Option<GcBox>>,

    /// Does the root needs to be traced?
    /// This should be `true` at the beginning of `Phase::Mark`.
    root_needs_trace: bool,

    /// A queue of gray objects, used during `Phase::Mark`.
    /// This holds traceable objects that have yet to be traced.
    gray: Cell<Option<GcBox>>,

    // A queue of gray objects that became gray as a result
    // of a write barrier.
    gray_again: Cell<Option<GcBox>>,
}

impl Context {
    pub(crate) unsafe fn drop(&mut self, a: &impl Allocator) {
        struct DropAll<'a, A: Allocator>(&'a Metrics, Option<GcBox>, &'a A);

        impl<'a, A: Allocator> Drop for DropAll<'a, A> {
            fn drop(&mut self) {
                if let Some(gc_box) = self.1.take() {
                    let mut drop_resume = DropAll(self.0, Some(gc_box), self.2);
                    while let Some(mut gc_box) = drop_resume.1.take() {
                        let header = gc_box.header();
                        drop_resume.1 = header.next();
                        let gc_size = gc_box.size_of_box();
                        // SAFETY: the context owns its GC'd objects
                        unsafe {
                            if header.is_live() {
                                gc_box.drop_in_place();
                                self.0.mark_gc_dropped(gc_size);
                            }
                            gc_box.dealloc(self.2);
                            self.0.mark_gc_freed(gc_size);
                        }
                    }
                }
            }
        }

        let cx = PhaseGuard::enter(self, Some(Phase::Drop));
        DropAll(&cx.metrics, cx.all.get(), a);
    }

    pub(crate) unsafe fn new() -> Context {
        let metrics = Metrics::new();
        Context {
            phase: Phase::Sleep,
            #[cfg(feature = "tracing")]
            phase_span: PhaseGuard::span_for(&metrics, Phase::Sleep),
            metrics: metrics.clone(),
            all: Cell::new(None),
            sweep: None,
            sweep_prev: Cell::new(None),
            root_needs_trace: true,
            gray: Cell::new(None),
            gray_again: Cell::new(None),
        }
    }

    #[inline]
    pub(crate) unsafe fn mutation_context<'gc>(&self, a: &dyn Allocator) -> Mutation<'gc> {
        // SAFETY: `Mutation` is a transparent wrapper around `Context`.
        unsafe {
            Mutation {
                context: mem::transmute::<&Self, &'gc Self>(self),
                alloc: mem::transmute::<&dyn Allocator, &'gc dyn Allocator>(a),
                _invariant: PhantomData,
            }
        }
    }

    #[inline]
    pub(crate) unsafe fn finalization_context<'gc>(&self, a: &dyn Allocator) -> Finalization<'gc> {
        // SAFETY: `Finalization` is a transparent wrapper around `Context`.
        unsafe { Finalization(self.mutation_context(a)) }
    }

    #[inline]
    pub(crate) fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    #[inline]
    pub(crate) fn root_barrier(&mut self) {
        if self.phase == Phase::Mark {
            self.root_needs_trace = true;
        }
    }

    #[inline]
    pub(crate) fn phase(&self) -> Phase {
        self.phase
    }

    #[inline]
    pub(crate) fn gray_remaining(&self) -> bool {
        self.gray.get().is_some() || self.gray_again.get().is_some() || self.root_needs_trace
    }

    // Do some collection work until either we have achieved our `target` (paying off debt or
    // finishing a full collection) or we have reached the `stop` condition.
    //
    // In order for this to be safe, at the time of call no `Gc` pointers can be live that are not
    // reachable from the given root object.
    //
    // If we are currently in `Phase::Sleep` and have positive debt, this will immediately
    // transition the collector to `Phase::Mark`.
    #[deny(unsafe_op_in_unsafe_fn)]
    pub(crate) unsafe fn do_collection<'gc, R: Collect<'gc> + ?Sized>(
        &mut self,
        root: &R,
        run_until: RunUntil,
        stop: Stop,
        a: &dyn Allocator,
    ) {
        let mut cx = PhaseGuard::enter(self, None);

        let debt = cx.metrics.allocation_debt();
        if run_until == RunUntil::PayDebt && !matches!(debt.partial_cmp(&0.0), Some(Greater)) {
            cx.log_progress("GC: paused");
            return;
        }

        let mut has_slept = false;

        loop {
            match cx.phase {
                Phase::Sleep => {
                    has_slept = true;
                    // Immediately enter the mark phase
                    cx.switch(Phase::Mark);
                }
                Phase::Mark => {
                    if cx.mark_one(root).is_break() {
                        if stop <= Stop::FullyMarked {
                            break;
                        } else {
                            // If we have no gray objects left, we enter the sweep phase.
                            cx.switch(Phase::Sweep);

                            // Set `sweep to the current head of our `all` linked list. Any new
                            // allocations during the newly-entered `Phase:Sweep` will update `all`,
                            // but will *not* be reachable from `this.sweep`.
                            cx.sweep = cx.all.get();
                        }
                    }
                }
                Phase::Sweep => {
                    if stop <= Stop::AtSweep {
                        break;
                    } else if cx.sweep_one(a).is_break() {
                        // Begin a new cycle.
                        //
                        // We reset our debt if we have done an entire collection cycle (marking and
                        // sweeping) as a single atomic unit. This keeps inherited debt from growing
                        // without bound.
                        cx.metrics.finish_cycle(has_slept);
                        cx.root_needs_trace = true;
                        cx.switch(Phase::Sleep);

                        // We treat a stop condition of `Stop::Finish` as special for the purposes
                        // of logging, and log that we finished a cycle.
                        if stop == Stop::FinishCycle {
                            cx.log_progress("GC: finished");
                            return;
                        }

                        // Otherwise we always break if we have performed a full cycle as a single
                        // atomic unit, because there cannot be any more work to do in this case.
                        if has_slept {
                            // We shouldn't be stopping here if the stop condition is something like
                            // `Stop::AtSweep`, but this should be impossible since the only way to
                            // get here is to have gone through the entire cycle.
                            assert!(stop == Stop::Full);
                            break;
                        }
                    }
                }
                Phase::Drop => unreachable!(),
            }

            let debt = cx.metrics.allocation_debt();
            if run_until == RunUntil::PayDebt && !matches!(debt.partial_cmp(&0.0), Some(Greater)) {
                break;
            }
        }

        cx.log_progress("GC: yielding...");
    }

    fn allocate<'gc, 'a, T, M, const ZEROED: bool>(&self, metadata: M, a: &dyn Allocator) -> GcBox
    where
        T: 'a + ?Sized,
        M: MetaCollect<'gc, 'a, T>,
    {
        let header = GcBoxHeader::new::<T, M>();
        header.set_next(self.all.get());
        header.set_live(true);
        header.set_needs_trace(metadata.needs_trace());

        let (alloc_layout, offset) =
            GcBox::box_layout::<T, M>(metadata).expect("layout calculation failed");

        let gc_box = unsafe {
            let mem = if ZEROED {
                a.allocate_zeroed(alloc_layout)
            } else {
                a.allocate(alloc_layout)
            };
            let Ok(mem) = mem else {
                ::alloc::alloc::handle_alloc_error(alloc_layout);
            };

            let ptr = mem.byte_add(offset - size_of::<GcBoxHeader>() - size_of::<M>());
            ptr.cast::<M>().write(metadata);

            let ptr = mem.byte_add(offset - size_of::<GcBoxHeader>());
            ptr.cast::<GcBoxHeader>().write(header);

            let ptr = mem.byte_add(offset);
            GcBox::from_raw(ptr.cast())
        };

        self.all.set(Some(gc_box));
        if self.phase == Phase::Sweep && self.sweep_prev.get().is_none() {
            self.sweep_prev.set(self.all.get());
        }

        self.metrics.mark_gc_allocated(alloc_layout.size());

        gc_box
    }

    #[inline]
    fn backward_barrier(&self, parent: GcBox, child: Option<GcBox>) {
        // During the marking phase, if we are mutating a black object, we may add a white object to
        // it and invalidate the invariant that black objects may not point to white objects. Turn
        // the black parent object gray to prevent this.
        //
        // NOTE: This also adds the pointer to the gray_again queue even if `header.needs_trace()`
        // is false, but this is not harmful (just wasteful). There's no reason to call a barrier on
        // a pointer that can't adopt other pointers, so we skip the check.
        if self.phase == Phase::Mark
            && parent.header().color() == GcColor::Black
            && child
                .map(|c| matches!(c.header().color(), GcColor::White | GcColor::WhiteWeak))
                .unwrap_or(true)
        {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, parent: GcBox) {
                this.make_gray_again(parent);
            }
            barrier(self, parent);
        }
    }

    #[inline]
    fn backward_barrier_weak(&self, parent: GcBox, child: GcBox) {
        if self.phase == Phase::Mark
            && parent.header().color() == GcColor::Black
            && child.header().color() == GcColor::White
        {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, parent: GcBox) {
                this.make_gray_again(parent);
            }
            barrier(self, parent);
        }
    }

    #[inline]
    fn forward_barrier(&self, parent: Option<GcBox>, child: GcBox) {
        // During the marking phase, if we are mutating a black object, we may add a white object
        // to it and invalidate the invariant that black objects may not point to white objects.
        // Immediately trace the child white object to turn it gray (or black) to prevent this.
        if self.phase == Phase::Mark
            && parent
                .map(|p| p.header().color() == GcColor::Black)
                .unwrap_or(true)
        {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, child: GcBox) {
                this.trace(child);
            }
            barrier(self, child);
        }
    }

    #[inline]
    fn forward_barrier_weak(&self, parent: Option<GcBox>, child: GcBox) {
        // During the marking phase, if we are mutating a black object, we may add a white object
        // to it and invalidate the invariant that black objects may not point to white objects.
        // Immediately trace the child white object to turn it gray (or black) to prevent this.
        if self.phase == Phase::Mark
            && parent
                .map(|p| p.header().color() == GcColor::Black)
                .unwrap_or(true)
        {
            // Outline the actual barrier code (which is somewhat expensive and won't be executed
            // often) to promote the inlining of the write barrier.
            #[cold]
            fn barrier(this: &Context, child: GcBox) {
                this.trace_weak(child);
            }
            barrier(self, child);
        }
    }

    #[inline]
    fn trace(&self, gc_box: GcBox) {
        let header = gc_box.header();
        let color = header.color();
        match color {
            GcColor::Black | GcColor::Gray => {}
            GcColor::White | GcColor::WhiteWeak => {
                if header.needs_trace() {
                    // A white traceable object is not in the gray queue, becomes gray and enters
                    // the normal gray queue.
                    header.set_color(GcColor::Gray);
                    debug_assert!(header.is_live());
                    header.set_gray_next(self.gray.get());
                    self.gray.set(Some(gc_box));
                } else {
                    // A white object that doesn't need tracing simply becomes black.
                    header.set_color(GcColor::Black);
                }

                // Only marking the *first* time counts as a mark metric.
                if color == GcColor::White {
                    self.metrics.mark_gc_marked(gc_box.size_of_box());
                }
            }
        }
    }

    #[inline]
    fn trace_weak(&self, gc_box: GcBox) {
        let header = gc_box.header();
        if header.color() == GcColor::White {
            header.set_color(GcColor::WhiteWeak);
            self.metrics.mark_gc_marked(gc_box.size_of_box());
        }
    }

    /// Determines whether or not a Gc pointer is safe to be upgraded.
    /// This is used by weak pointers to determine if it can safely upgrade to a strong pointer.
    #[inline]
    fn upgrade(&self, gc_box: GcBox) -> bool {
        let header = gc_box.header();

        // This object has already been freed, definitely not safe to upgrade.
        if !header.is_live() {
            return false;
        }

        // Consider the different possible phases of the GC:
        // * In `Phase::Sleep`, the GC is not running, so we can upgrade.
        //   If the newly-created `Gc` or `GcCell` survives the current `arena.mutate`
        //   call, then the situtation is equivalent to having copied an existing `Gc`/`GcCell`,
        //   or having created a new allocation.
        //
        // * In `Phase::Mark`:
        //   If the newly-created `Gc` or `GcCell` survives the current `arena.mutate`
        //   call, then it must have been stored somewhere, triggering a write barrier.
        //   This will ensure that the new `Gc`/`GcCell` gets traced (if it's now reachable)
        //   before we transition to `Phase::Sweep`.
        //
        // * In `Phase::Sweep`:
        //   If the allocation is `WhiteWeak`, then it's impossible for it to have been freshly-
        //   created during this `Phase::Sweep`. `WhiteWeak` is only  set when a white `Weak/
        //   WeakCell` is traced. A `Weak/WeakCell` must be created from an existing `Gc/
        //   GcCell` via `downgrade()`, so `WhiteWeak` means that a `Weak` / `WeakCell` existed
        //   during the last `Phase::Mark.`
        //
        //   Therefore, a `WhiteWeak` object is guaranteed to be deallocated during this
        //   `Phase::Sweep`, and we must not upgrade it.
        //
        //   Conversely, it's always safe to upgrade a white object that is not `WhiteWeak`.
        //   In order to call `upgrade`, you must have a `Weak/WeakCell`. Since it is
        //   not `WhiteWeak` there cannot have been any `Weak/WeakCell`s during the
        //   last `Phase::Mark`, so the weak pointer must have been created during this
        //   `Phase::Sweep`. This is only possible if the underlying allocation was freshly-created
        //   - if the allocation existed during `Phase::Mark` but was not traced, then it
        //   must have been unreachable, which means that the user wouldn't have been able to call
        //   `downgrade`. Therefore, we can safely upgrade, knowing that the object will not be
        //   freed during this phase, despite being white.
        if self.phase == Phase::Sweep && header.color() == GcColor::WhiteWeak {
            return false;
        }
        true
    }

    #[inline]
    fn resurrect(&self, gc_box: GcBox) {
        let header = gc_box.header();
        debug_assert_eq!(self.phase, Phase::Mark);
        debug_assert!(header.is_live());
        let color = header.color();
        if matches!(header.color(), GcColor::White | GcColor::WhiteWeak) {
            header.set_color(GcColor::Gray);
            header.set_gray_next(self.gray.get());
            self.gray.set(Some(gc_box));
            // Only marking the *first* time counts as a mark metric.
            if color == GcColor::White {
                self.metrics.mark_gc_marked(gc_box.size_of_box());
            }
        }
    }

    fn mark_one<'gc, R: Collect<'gc> + ?Sized>(&mut self, root: &R) -> ControlFlow<()> {
        // We look for an object first in the normal gray queue, then the "gray again" queue.
        // Processing "gray again" objects later gives them more time to be mutated again without
        // triggering another write barrier.
        let pop = |list: &Cell<Option<GcBox>>| {
            list.get().inspect(|gc_box| {
                list.set(gc_box.header().gray_next());
                gc_box.header().set_gray_next(None);
            })
        };
        let next_gray = pop(&self.gray).or_else(|| pop(&self.gray_again));

        if let Some(gc_box) = next_gray {
            // We always mark work for objects processed from both the gray and "gray again" queue.
            // When objects are placed into the "gray again" queue due to a write barrier, the
            // original work is *undone*, so we do it again here.
            self.metrics.mark_gc_traced(gc_box.size_of_box());
            gc_box.header().set_color(GcColor::Black);

            // If we have an object in the gray queue, take one, trace it, and turn it black.

            // Our `Collect::trace` call may panic, and if it does the object will be lost from
            // the gray queue but potentially incompletely traced. By catching a panic during
            // `Arena::collect()`, this could lead to memory unsafety.
            //
            // So, if the `Collect::trace` call panics, we need to add the popped object back to the
            // `gray_again` queue. If the panic is caught, this will maybe give some time for its
            // trace method to not panic before attempting to collect it again.
            struct DropGuard<'a> {
                context: &'a mut Context,
                gc_box: GcBox,
            }

            impl<'a> Drop for DropGuard<'a> {
                fn drop(&mut self) {
                    self.context.make_gray_again(self.gc_box);
                }
            }

            let guard = DropGuard {
                context: self,
                gc_box,
            };
            debug_assert!(gc_box.header().is_live());
            unsafe { gc_box.trace_value(guard.context) }
            mem::forget(guard);

            ControlFlow::Continue(())
        } else if self.root_needs_trace {
            // We treat the root object as gray if `root_needs_trace` is set, and we process it at
            // the end of the gray queue for the same reason as the "gray again" objects.
            root.trace(self);
            self.root_needs_trace = false;
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    }

    fn sweep_one(&mut self, a: &dyn Allocator) -> ControlFlow<()> {
        let Some(mut sweep) = self.sweep else {
            self.sweep_prev.set(None);
            return ControlFlow::Break(());
        };

        let sweep_header = sweep.header();
        let sweep_size = sweep.size_of_box();

        let next_box = sweep_header.next();
        self.sweep = next_box;

        match sweep_header.color() {
            // If the next object in the sweep portion of the main list is white, we
            // need to remove it from the main object list and destruct it.
            GcColor::White => {
                if let Some(sweep_prev) = self.sweep_prev.get() {
                    sweep_prev.header().set_next(next_box);
                } else {
                    // If `sweep_prev` is None, then the sweep pointer is also the
                    // beginning of the main object list, so we need to adjust it.
                    debug_assert_eq!(self.all.get(), Some(sweep));
                    self.all.set(next_box);
                }

                // SAFETY: this object is white, and wasn't traced by a `Weak` during this cycle,
                // meaning it cannot have either strong or weak pointers, so we can drop the whole
                // object.
                unsafe {
                    if sweep_header.is_live() {
                        // If the alive flag is set, that means we haven't dropped the inner value
                        // of this object,
                        sweep.drop_in_place();
                        self.metrics.mark_gc_dropped(sweep_size);
                    }
                    sweep.dealloc(a);
                    self.metrics.mark_gc_freed(sweep_size);
                }
            }
            // Keep the `GcBox` as part of the linked list if we traced a weak pointer to it. The
            // weak pointer still needs access to the `GcBox` to be able to check if the object
            // is still alive. We can only deallocate the `GcBox`, once there are no weak pointers
            // left.
            GcColor::WhiteWeak => {
                self.sweep_prev.set(Some(sweep));
                sweep_header.set_color(GcColor::White);
                if sweep_header.is_live() {
                    sweep_header.set_live(false);
                    // SAFETY: Since this object is white, that means there are no more strong
                    // pointers to this object, only weak pointers, so we can safely drop its
                    // contents.
                    unsafe { sweep.drop_in_place() }
                    self.metrics.mark_gc_dropped(sweep_size);
                }
                self.metrics.mark_gc_remembered(sweep_size);
            }
            // If the next object in the sweep portion of the main list is black, we
            // need to keep it but turn it back white.
            GcColor::Black => {
                self.sweep_prev.set(Some(sweep));
                sweep_header.set_color(GcColor::White);
                self.metrics.mark_gc_remembered(sweep_size);
            }
            // No gray objects should be in this part of the main list, they should
            // be added to the beginning of the list before the sweep pointer, so it
            // should not be possible for us to encounter them here.
            GcColor::Gray => {
                debug_assert!(false, "unexpected gray object in sweep list")
            }
        }

        ControlFlow::Continue(())
    }

    // Take a black pointer and turn it gray and put it in the `gray_again` queue.
    fn make_gray_again(&self, gc_box: GcBox) {
        let header = gc_box.header();
        debug_assert_eq!(header.color(), GcColor::Black);
        header.set_color(GcColor::Gray);
        header.set_gray_next(self.gray_again.get());
        self.gray_again.set(Some(gc_box));
        self.metrics.mark_gc_untraced(gc_box.size_of_box());
    }
}

/// Helper type for managing phase transitions.
struct PhaseGuard<'a> {
    cx: &'a mut Context,
    #[cfg(feature = "tracing")]
    span: tracing::span::EnteredSpan,
}

impl<'a> Drop for PhaseGuard<'a> {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        {
            let span = mem::replace(&mut self.span, tracing::Span::none().entered());
            self.cx.phase_span = span.exit();
        }
    }
}

impl<'a> Deref for PhaseGuard<'a> {
    type Target = Context;

    #[inline(always)]
    fn deref(&self) -> &Context {
        self.cx
    }
}

impl<'a> DerefMut for PhaseGuard<'a> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Context {
        self.cx
    }
}

impl<'a> PhaseGuard<'a> {
    fn enter(cx: &'a mut Context, phase: Option<Phase>) -> Self {
        if let Some(phase) = phase {
            cx.phase = phase;
        }

        Self {
            #[cfg(feature = "tracing")]
            span: {
                let mut span = mem::replace(&mut cx.phase_span, tracing::Span::none());
                if let Some(phase) = phase {
                    span = Self::span_for(&cx.metrics, phase);
                }
                span.entered()
            },
            cx,
        }
    }

    fn switch(&mut self, phase: Phase) {
        self.cx.phase = phase;

        #[cfg(feature = "tracing")]
        {
            let _ = mem::replace(&mut self.span, tracing::Span::none().entered());
            self.span = Self::span_for(&self.cx.metrics, phase).entered();
        }
    }

    fn log_progress(&mut self, #[allow(unused)] message: &str) {
        // TODO: add more infos here
        #[cfg(feature = "tracing")]
        tracing::debug!(
            target: "gc_arena",
            parent: &self.span,
            message,
            phase = tracing::field::debug(self.cx.phase),
            allocated = self.cx.metrics.total_allocation(),
        );
    }

    #[cfg(feature = "tracing")]
    fn span_for(metrics: &Metrics, phase: Phase) -> tracing::Span {
        tracing::debug_span!(
            target: "gc_arena",
            "gc_arena",
            id = metrics.arena_id(),
            ?phase,
        )
    }
}
