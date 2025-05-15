#![allow(clippy::assertions_on_constants)]
#![allow(clippy::redundant_closure)]

#[cfg(feature = "std")]
use std::rc::Rc;

use gc_arena::{Arena, Collect, Gc, Rootable, gc::Unique, static_collect};

#[test]
fn simple_allocation() {
    #[derive(Collect)]
    #[collect(no_drop)]
    struct TestRoot<'gc> {
        test: Unique<'gc, i32>,
    }

    let arena = Arena::<Rootable![TestRoot<'_>]>::new(|mc| TestRoot {
        test: Unique::new(mc, 42),
    });

    arena.mutate(|_mc, root| {
        assert_eq!(*(root.test), 42);
    });
}

#[cfg(feature = "std")]
#[test]
fn dyn_sized_allocation() {
    #[derive(Clone)]
    struct RefCounter(Rc<()>);
    static_collect!(RefCounter);

    #[derive(Collect)]
    #[collect(no_drop)]
    struct TestRoot<'gc> {
        slice: Gc<'gc, [Unique<'gc, RefCounter>]>,
    }

    const SIZE: usize = 10;

    let counter = RefCounter(Rc::new(()));

    let mut arena = Arena::<Rootable![TestRoot<'_>]>::new(|mc| {
        let array: [_; SIZE] = core::array::from_fn(|_| Unique::new(mc, counter.clone()));
        let slice = Gc::new_unsize(mc, array);
        TestRoot { slice }
    });

    arena.finish_cycle();

    // Check that no counter was dropped.
    assert_eq!(Rc::strong_count(&counter.0), SIZE + 1);

    // Drop all the RefCounters.
    arena.mutate_root(|mc, root| {
        root.slice = Gc::new_unsize(mc, []);
    });
    arena.finish_cycle();

    // Check that all counters were dropped.
    assert_eq!(Rc::strong_count(&counter.0), 1);
}
