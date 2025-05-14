#![no_std]
#![feature(derive_coerce_pointee)]
#![feature(layout_for_ptr)]
#![feature(maybe_uninit_write_slice)]
#![feature(ptr_metadata)]
#![cfg_attr(miri, feature(maybe_uninit_as_bytes))]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

pub mod arena;
pub mod barrier;
pub mod collect;
mod collect_impl;
mod context;
pub mod dynamic_roots;
mod gc;
mod gc_weak;
pub mod lock;
pub mod metrics;
mod no_drop;
mod static_collect;
mod types;
mod unique_gc;

#[cfg(feature = "allocator-api2")]
pub mod allocator_api;

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
    gc_weak::GcWeak,
    lock::{GcLock, GcRefLock, Lock, RefLock},
    static_collect::Static,
    unique_gc::UniqueGc,
};
