#![no_std]
#![feature(allocator_api)]
#![feature(associated_type_defaults)]
#![feature(layout_for_ptr)]
#![feature(maybe_uninit_write_slice, maybe_uninit_fill, maybe_uninit_slice)]
#![feature(min_specialization)]
#![feature(ptr_metadata)]
#![feature(trusted_len)]
#![feature(unsize)]
#![cfg_attr(miri, feature(maybe_uninit_as_bytes))]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod arena;
pub mod barrier;
pub mod collect;
mod context;
pub mod dynamic_roots;
pub mod gc;
pub mod lock;
pub mod metrics;
mod no_drop;
pub mod ptr;
mod types;
pub mod vec;

#[cfg(feature = "hashbrown")]
mod hashbrown;

#[doc(hidden)]
pub use gc_arena_derive::__unelide_lifetimes;

#[doc(hidden)]
pub use self::{arena::__DynRootable, no_drop::__MustNotImplDrop};

pub use self::{
    arena::{Arena, Rootable},
    collect::Collect,
    context::{Finalization, Mutation},
    dynamic_roots::{DynamicRoot, DynamicRootSet},
    gc::Gc,
    lock::{GcLock, GcRefLock, Lock, RefLock},
};
