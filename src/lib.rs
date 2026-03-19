#![no_std]
#![feature(allocator_api)]
#![feature(maybe_uninit_fill)]
#![feature(min_specialization)]
#![feature(ptr_metadata)]
#![feature(trusted_len)]
#![feature(unsize)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod arena;
pub mod barrier;
pub mod collect;
mod context;
pub mod gc;
pub mod lock;
pub mod metrics;
mod no_drop;
pub mod ptr;
pub mod root;
mod types;
pub mod vec;

#[cfg(feature = "hashbrown")]
mod hashbrown;

mod sealed {
    /// Forbids implementing some trait for downstream types to ensure
    /// soundness.
    pub trait Sealed {}
}

#[doc(hidden)]
pub use gc_arena_derive::__unelide_lifetimes;

#[doc(hidden)]
pub use self::{arena::__DynRootable, no_drop::__MustNotImplDrop};
pub use self::{
    arena::{Arena, Rootable},
    collect::Collect,
    context::{Finalization, Mutation},
    gc::Gc,
    lock::{GcLock, GcRefLock, Lock, RefLock},
};
