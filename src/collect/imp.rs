use alloc::{
    alloc::Global,
    boxed::Box,
    collections::{BTreeMap, BTreeSet, BinaryHeap, LinkedList, VecDeque},
    rc::Rc,
    string::String,
    vec::Vec,
};
use core::{
    alloc::Allocator,
    cell::{Cell, RefCell},
    marker::PhantomData,
    mem::MaybeUninit,
};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

use crate::collect::{Collect, CollectRef, Trace};

unsafe impl<'gc, T: CollectRef<'gc> + ?Sized> Collect<'gc> for &T {
    const NEEDS_TRACE: bool = <T as CollectRef<'gc>>::NEEDS_TRACE;

    #[inline]
    fn trace<C: Trace<'gc>>(&mut self, cc: &mut C) {
        <T as CollectRef<'gc>>::trace_ref(self, cc)
    }
}

/// If a type is static, we know that it can never hold `Gc` pointers, so it is
/// safe to provide a simple empty `Collect` implementation.
#[macro_export]
macro_rules! static_collect {
    ($type:ty) => {
        unsafe impl<'gc> $crate::collect::Collect<'gc> for $type
        where
            $type: 'static,
        {
            const NEEDS_TRACE: bool = false;
        }

        unsafe impl<'gc> $crate::collect::CollectRef<'gc> for $type
        where
            $type: 'static,
        {
            const NEEDS_TRACE: bool = false;
        }
    };
}

static_collect!(bool);
static_collect!(char);
static_collect!(u8);
static_collect!(u16);
static_collect!(u32);
static_collect!(u64);
static_collect!(usize);
static_collect!(i8);
static_collect!(i16);
static_collect!(i32);
static_collect!(i64);
static_collect!(isize);
static_collect!(f32);
static_collect!(f64);
static_collect!(String);
static_collect!(str);
static_collect!(alloc::ffi::CString);
static_collect!(core::ffi::CStr);
static_collect!(core::any::TypeId);
#[cfg(feature = "std")]
static_collect!(std::path::Path);
#[cfg(feature = "std")]
static_collect!(std::path::PathBuf);
#[cfg(feature = "std")]
static_collect!(std::ffi::OsStr);
#[cfg(feature = "std")]
static_collect!(std::ffi::OsString);

/// For the purposes of tracing, a `MaybeUninit` is assumed to always be
/// uninitialized. This means that the collector will never actually trace its
/// contents. Therefore it will likely cause undefined behaviour to read a
/// garbage collected pointer from the `MaybeUninit`, if it was set in a prior
/// mutation.
unsafe impl<'gc, T> CollectRef<'gc> for MaybeUninit<T> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc, T> Collect<'gc> for MaybeUninit<T> {
    const NEEDS_TRACE: bool = false;
}

static_collect!(Global);

macro_rules! forward_collect {
    (
        $(#[$m:meta])*
        $type:ty = ($($params:tt)*) where ($($bounds:tt)*) / ($($bounds_ref:tt)*) {
            const $NEEDS_TRACE:ident: bool = $needs_trace:expr;
            fn trace = |$self:ident, $cc:ident|
            $trace:block / $trace_ref:block
        }
    ) => {
        $(#[$m])*
        unsafe impl<'gc, $($params)*> $crate::collect::Collect<'gc> for $type
        where
            $($bounds)*
        {
            const $NEEDS_TRACE: bool = $needs_trace;

            #[inline]
            fn trace<Cc: $crate::collect::Trace<'gc>>(&mut self, cc: &mut Cc) {
                (|$self: &mut Self, $cc: &mut Cc| $trace)(self, cc)
            }
        }

        $(#[$m])*
        unsafe impl<'gc, $($params)*> $crate::collect::CollectRef<'gc> for $type
        where
            $($bounds_ref)*
        {
            const $NEEDS_TRACE: bool = $needs_trace;

            #[inline]
            fn trace_ref<Cc: $crate::collect::Trace<'gc>>(&self, cc: &mut Cc) {
                (|$self: &Self, $cc: &mut Cc| $trace_ref)(self, cc)
            }
        }
    }
}

forward_collect! {
    Box<T, A> = (T: ?Sized, A: Allocator + CollectRef<'gc>)
        where (T: Collect<'gc>) / (T: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            cc.trace(&mut **this);
            cc.trace_ref(Box::allocator(this));
        } / {
            cc.trace_ref(&**this);
            cc.trace_ref(Box::allocator(this));
        }
    }
}

forward_collect! {
    [T] = (T) where (T: Collect<'gc>) / (T: CollectRef<'gc>) {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter_mut().for_each(|t| cc.trace(t));
        } / {
            this.iter().for_each(|t| cc.trace_ref(t));
        }
    }
}

forward_collect! {
    Option<T> = (T) where (T: Collect<'gc>) / (T: CollectRef<'gc>) {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            if let Some(t) = this.as_mut() {
                cc.trace(t);
            }
        } / {
            if let Some(t) = this.as_ref() {
                cc.trace_ref(t);
            }
        }
    }
}

forward_collect! {
    Result<T, E> = (T, E)
        where (T: Collect<'gc>, E: Collect<'gc>) / (T: CollectRef<'gc>, E: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || E::NEEDS_TRACE;
        fn trace = |this, cc| {
            match this {
                Ok(r) => cc.trace(r),
                Err(e) => cc.trace(e),
            }
        } / {
            match this {
                Ok(r) => cc.trace_ref(r),
                Err(e) => cc.trace_ref(e),
            }
        }
    }
}

forward_collect! {
    Vec<T, A> = (T, A: Allocator + CollectRef<'gc>)
        where (T: Collect<'gc>) / (T: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter_mut().for_each(|t| cc.trace(t));
            cc.trace_ref(this.allocator());
        } / {
            this.iter().for_each(|t| cc.trace_ref(t));
            cc.trace_ref(this.allocator());
        }
    }
}

forward_collect! {
    VecDeque<T, A> = (T, A: Allocator + CollectRef<'gc>)
        where (T: Collect<'gc>) / (T: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter_mut().for_each(|t| cc.trace(t));
            cc.trace_ref(this.allocator());
        } / {
            this.iter().for_each(|t| cc.trace_ref(t));
            cc.trace_ref(this.allocator());
        }
    }
}

#[cfg(feature = "std")]
forward_collect! {
    HashMap<K, V, S> = (K: CollectRef<'gc>, V, S: 'static)
        where (V: Collect<'gc>) / (V: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = K::NEEDS_TRACE || V::NEEDS_TRACE;
        fn trace = |this, cc| {
            for (k, v) in this {
                cc.trace_ref(k);
                cc.trace(v);
            }
        } / {
            for (k, v) in this {
                cc.trace_ref(k);
                cc.trace_ref(v);
            }
        }
    }
}

#[cfg(feature = "std")]
forward_collect! {
    HashSet<T, S> = (T: CollectRef<'gc>, S: 'static) where () / () {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter().for_each(|v| cc.trace_ref(v));
        } / {
            this.iter().for_each(|v| cc.trace_ref(v));
        }
    }
}

forward_collect! {
    BinaryHeap<T, A> = (T: CollectRef<'gc>, A: Allocator + CollectRef<'gc>)
        where () / ()
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter().for_each(|v| cc.trace_ref(v));
            cc.trace_ref(this.allocator());
        } / {
            this.iter().for_each(|v| cc.trace_ref(v));
            cc.trace_ref(this.allocator());
        }
    }
}

// FIXME: Add allocator tracing for `alloc::collections::*` once their APIs are
// exposed.

forward_collect! {
    LinkedList<T> = (T) where (T: Collect<'gc>) / (T: CollectRef<'gc>) {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter_mut().for_each(|t| cc.trace(t));
        } / {
            this.iter().for_each(|t| cc.trace_ref(t));
        }
    }
}

forward_collect! {
    BTreeMap<K, V> = (K: CollectRef<'gc>, V)
        where (V: Collect<'gc>) / (V: CollectRef<'gc>)
    {
        const NEEDS_TRACE: bool = K::NEEDS_TRACE || V::NEEDS_TRACE;
        fn trace = |this, cc| {
            for (k, v) in this {
                cc.trace_ref(k);
                cc.trace(v);
            }
        } / {
            for (k, v) in this {
                cc.trace_ref(k);
                cc.trace_ref(v);
            }
        }
    }
}

forward_collect! {
    BTreeSet<T> = (T: CollectRef<'gc>) where () / () {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter().for_each(|v| cc.trace_ref(v));
        } / {
            this.iter().for_each(|v| cc.trace_ref(v));
        }
    }
}

forward_collect! {
    Rc<T, A> = (T: CollectRef<'gc>, A: Allocator + CollectRef<'gc>)
        where () / ()
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            cc.trace_ref(&**this);
            cc.trace_ref(Rc::allocator(this));
        } / {
            cc.trace_ref(&**this);
            cc.trace_ref(Rc::allocator(this));
        }
    }
}

#[cfg(target_has_atomic = "ptr")]
forward_collect! {
    alloc::sync::Arc<T, A> = (T: CollectRef<'gc>, A: Allocator + CollectRef<'gc>)
        where () / ()
    {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE || A::NEEDS_TRACE;
        fn trace = |this, cc| {
            cc.trace_ref(&**this);
            cc.trace_ref(alloc::sync::Arc::allocator(this));
        } / {
            cc.trace_ref(&**this);
            cc.trace_ref(alloc::sync::Arc::allocator(this));
        }
    }
}

unsafe impl<'gc, T: 'static> Collect<'gc> for Cell<T> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc, T: 'static> CollectRef<'gc> for Cell<T> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc, T: 'static> Collect<'gc> for RefCell<T> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc, T: 'static> CollectRef<'gc> for RefCell<T> {
    const NEEDS_TRACE: bool = false;
}

// SAFETY: `PhantomData` is a ZST, and therefore doesn't store anything
unsafe impl<'gc, T: ?Sized> Collect<'gc> for PhantomData<T> {
    const NEEDS_TRACE: bool = false;
}

unsafe impl<'gc, T: ?Sized> CollectRef<'gc> for PhantomData<T> {
    const NEEDS_TRACE: bool = false;
}

forward_collect! {
    [T; N] = (T, const N: usize) where (T: Collect<'gc>) / (T: CollectRef<'gc>) {
        const NEEDS_TRACE: bool = T::NEEDS_TRACE;
        fn trace = |this, cc| {
            this.iter_mut().for_each(|t| cc.trace(t));
        } / {
            this.iter().for_each(|t| cc.trace_ref(t));
        }
    }
}

macro_rules! impl_tuple {
    () => (
        unsafe impl<'gc> Collect<'gc> for () {
            const NEEDS_TRACE: bool = false;
        }

        unsafe impl<'gc> CollectRef<'gc> for () {
            const NEEDS_TRACE: bool = false;
        }
    );

    ($($name:ident)+) => (
        forward_collect! {
            #[allow(unused_parens, non_snake_case)]
            (($($name,)*)) = ($($name,)*)
                where ($($name: Collect<'gc>,)*) / ($($name: CollectRef<'gc>,)*)
            {
                const NEEDS_TRACE: bool = false $(|| $name::NEEDS_TRACE)*;

                fn trace = |this, cc| {
                    let ($($name,)*) = this;
                    $(cc.trace($name);)*
                } / {
                    let ($($name,)*) = this;
                    $(cc.trace_ref($name);)*
                }
            }
        }
    );
}

impl_tuple! {}
impl_tuple! {A}
impl_tuple! {A B}
impl_tuple! {A B C}
impl_tuple! {A B C D}
impl_tuple! {A B C D E}
impl_tuple! {A B C D E F}
impl_tuple! {A B C D E F G}
impl_tuple! {A B C D E F G H}
impl_tuple! {A B C D E F G H I}
impl_tuple! {A B C D E F G H I J}
impl_tuple! {A B C D E F G H I J K}
impl_tuple! {A B C D E F G H I J K L}
impl_tuple! {A B C D E F G H I J K L M}
impl_tuple! {A B C D E F G H I J K L M N}
impl_tuple! {A B C D E F G H I J K L M N O}
impl_tuple! {A B C D E F G H I J K L M N O P}
