use core::{
    cmp, fmt,
    hash::{Hash, Hasher},
    iter::TrustedLen,
    mem::{ManuallyDrop, MaybeUninit},
    ops::{Deref, DerefMut, Index, IndexMut},
    ptr,
    slice::{self, SliceIndex},
};

use spec_from_iter::SpecFromIter;

use self::set_len_on_drop::SetLenOnDrop;
use crate::{Collect, Gc, Mutation, collect::Trace, gc::Unique};

mod convert_vec;
mod iter;
mod set_len_on_drop;
mod spec_extend;
mod spec_from_elem;
mod spec_from_iter;

pub use self::{convert_vec::ConvertVec, iter::IntoIter};
use self::{spec_extend::SpecExtend, spec_from_elem::SpecFromElem};

// Tiny Vecs are dumb. Skip to:
// - 8 if the element size is 1, because any heap allocators is likely to round
//   up a request of less than 8 bytes to at least 8 bytes.
// - 4 if elements are moderate-sized (<= 1 KiB).
// - 1 otherwise, to avoid wasting too much space for very short Vecs.
const fn min_non_zero_cap(size: usize) -> usize {
    if size == 1 {
        8
    } else if size <= 1024 {
        4
    } else {
        1
    }
}

/// A contiguous growable array type with GC-managed buffer.
pub struct Vec<'gc, T: 'gc> {
    buf: Unique<'gc, [MaybeUninit<T>]>,
    len: usize,
}

unsafe impl<'gc, T: 'gc + Collect<'gc>> Collect<'gc> for Vec<'gc, T> {
    const NEEDS_TRACE: bool = true;

    fn trace<U: Trace<'gc>>(&mut self, cc: &mut U) {
        cc.trace(&mut self.buf);
        cc.trace(self.deref_mut());
    }
}

/// Creates a GC'd [`Vec`] containing the arguments.
///
/// `vec!` allows `Vec`s to be defined with the same syntax as array
/// expressions. There are two forms of this macro:
///
/// - Create a [`Vec`] containing a given list of elements:
///
/// ```
/// # use gc_arena::{arena::rootless_mutate, vec};
/// # rootless_mutate(|mc| {
/// let v = vec![mc => 1, 2, 3];
/// assert_eq!(v[0], 1);
/// assert_eq!(v[1], 2);
/// assert_eq!(v[2], 3);
/// # });
/// ```
///
/// - Create a [`Vec`] from a given element and size:
///
/// ```
/// # use gc_arena::{arena::rootless_mutate, vec};
/// # rootless_mutate(|mc| {
/// let v = vec![mc => 1; 3];
/// assert_eq!(v, [1, 1, 1]);
/// # });
/// ```
///
/// Note that unlike array expressions this syntax supports all elements
/// which implement [`Clone`] and the number of elements doesn't have to be
/// a constant.
///
/// This will use `clone` to duplicate an expression, so one should be careful
/// using this with types having a nonstandard `Clone` implementation. For
/// example, `vec![Rc::new(1); 5]` will create a vector of five references
/// to the same boxed integer value, not five references pointing to
/// independently boxed integers.
///
/// Also, note that `vec![expr; 0]` is allowed, and produces an empty vector.
/// This will still evaluate `expr`, however, and immediately drop the resulting
/// value, so be mindful of side effects.
///
/// [`Vec`]: crate::vec::Vec
#[macro_export]
macro_rules! vec {
    [$mc:expr] => [$crate::vec::Vec::new($mc)];
    [$mc:expr => $elem:expr; $count:expr] => {
        $crate::vec::Vec::from_elem($mc, $elem, $count)
    };
    [$mc:expr => $($elem:expr),* $(,)?] => {
        $crate::vec::Vec::from_array($mc, [$($elem),*])
    };
}

impl<'gc, T: 'gc> Vec<'gc, T> {
    pub(crate) const ASSERT_NO_DROP: () = assert!(!core::mem::needs_drop::<T>());
}

impl<'gc, T: 'gc + Collect<'gc>> Vec<'gc, T> {
    /// Constructs a new, empty `Vec<'gc, T>`.
    ///
    /// Unlike [`Vec<T>`] from the standard library, this function **does
    /// allocate** a minimal amount of buffer space. This is because
    /// [`Gc<'gc, T>`] needs a header stored in the heap.
    ///
    /// [`Vec<T>`]: alloc::vec::Vec
    ///
    /// # Examples
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, vec::Vec};
    /// # rootless_mutate(|mc| {
    /// let mut vec = Vec::new(mc);
    /// vec.push(mc, 1);
    /// vec.push(mc, 2);
    ///
    /// assert_eq!(*vec, [1, 2]);
    /// assert_eq!(vec.pop(), Some(2));
    /// assert_eq!(*vec, [1]);
    /// # });
    /// ```
    pub fn new(mc: &Mutation<'gc>) -> Self {
        let () = Self::ASSERT_NO_DROP;
        Self::with_capacity(mc, min_non_zero_cap(size_of::<T>()))
    }

    /// Constructs a new, empty `Vec<'gc, T>` with at least the specified
    /// capacity.
    ///
    /// The vector will be able to hold at least `capacity` elements without
    /// reallocating. This method is allowed to allocate for more elements than
    /// `capacity`. If `capacity` is zero, the vector will not allocate.
    ///
    /// If it is important to know the exact allocated capacity of a `Vec`,
    /// always use the [`capacity`] method after construction.
    ///
    /// For `Vec<'gc, T>` where `T` is a zero-sized type, there will be no
    /// allocation and the capacity will always be `usize::MAX`.
    ///
    /// [`capacity`]: Vec::capacity
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    ///
    /// # Examples
    ///
    /// ```
    /// # use gc_arena::{arena::rootless_mutate, vec::Vec};
    /// # rootless_mutate(|mc| {
    /// let mut vec = Vec::with_capacity(mc, 10);
    ///
    /// // The vector contains no items, even though it has capacity for more
    /// assert_eq!(vec.len(), 0);
    /// assert!(vec.capacity() == 10);
    ///
    /// // These are all done without reallocating...
    /// for i in 0..10 {
    ///     vec.push(mc, i);
    /// }
    /// assert_eq!(vec.len(), 10);
    /// assert!(vec.capacity() == 10);
    ///
    /// // ...but this may make the vector reallocate
    /// vec.push(mc, 11);
    /// assert_eq!(vec.len(), 11);
    /// assert!(vec.capacity() >= 11);
    ///
    /// // A vector of a zero-sized type will always over-allocate, since no
    /// // allocation is necessary
    /// let vec_units = Vec::<()>::with_capacity(mc, 10);
    /// assert_eq!(vec_units.capacity(), usize::MAX);
    /// # });
    /// ```
    pub fn with_capacity(mc: &Mutation<'gc>, capacity: usize) -> Self {
        let () = Self::ASSERT_NO_DROP;
        let cap = if const { size_of::<T>() == 0 } {
            usize::MAX
        } else {
            capacity
        };
        let buf = Gc::new_uninit_slice(mc, cap);
        Self { buf, len: 0 }
    }

    /// Constructs a new `Vec<'gc, T>` from an array.
    #[inline]
    pub fn from_array<const N: usize>(mc: &Mutation<'gc>, array: [T; N]) -> Self {
        Unique::new_unsize::<[T]>(mc, array).into_vec()
    }

    /// # Safety
    ///
    /// `cap` must be greater than or equal to `self.len()`.
    #[cold]
    unsafe fn reallocate(&mut self, mc: &Mutation<'gc>, cap: usize) {
        let mut new_buf = Gc::new_uninit_slice(mc, cap);
        // SAFETY: `new_buf` is uninitialized and valid. The old buf will be discarded
        // and collected automatically without touching the old copies of elements.
        unsafe { ptr::copy_nonoverlapping(self.buf.as_ptr(), new_buf.as_mut_ptr(), self.len) }
        self.buf = new_buf;
    }

    /// Reserves capacity for at least `additional` more elements to be inserted
    /// in the given `Vec<T>`. The collection may reserve more space to
    /// speculatively avoid frequent reallocations. After calling `reserve`,
    /// capacity will be greater than or equal to `self.len() + additional`.
    /// Does nothing if capacity is already sufficient.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    pub fn reserve(&mut self, mc: &Mutation<'gc>, additional: usize) {
        if self.len + additional > self.buf.len() {
            debug_assert!(size_of::<T>() > 0);

            let required_cap = self.len + additional;

            let cap = cmp::max(self.buf.len() * 2, required_cap);
            let cap = cmp::max(min_non_zero_cap(size_of::<T>()), cap);

            // SAFETY: `cap` >= `self.len`.
            unsafe { self.reallocate(mc, cap) };
        }
    }

    /// Reserves the minimum capacity for at least `additional` more elements to
    /// be inserted in the given `Vec<T>`. Unlike [`reserve`], this will not
    /// deliberately over-allocate to speculatively avoid frequent allocations.
    /// After calling `reserve_exact`, capacity will be greater than or equal to
    /// `self.len() + additional`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity can not be relied upon to be precisely
    /// minimal. Prefer [`reserve`] if future insertions are expected.
    ///
    /// [`reserve`]: Vec::reserve
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    pub fn reserve_exact(&mut self, mc: &Mutation<'gc>, additional: usize) {
        if self.len + additional > self.buf.len() {
            debug_assert!(size_of::<T>() > 0);

            let required_cap = self.len + additional;

            // SAFETY: `required_cap` >= `self.len`.
            unsafe { self.reallocate(mc, required_cap) };
        }
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self, mc: &Mutation<'gc>) {
        if self.len < self.buf.len() && size_of::<T>() > 0 {
            // SAFETY: `cap` <- `self.len`.
            unsafe { self.reallocate(mc, self.len) };
        }
    }

    /// Shrinks the capacity of the vector with a lower bound.
    ///
    /// The capacity will remain at least as large as both the length
    /// and the supplied value.
    ///
    /// If the current capacity is less than the lower limit, this is a no-op.
    pub fn shrink_to(&mut self, mc: &Mutation<'gc>, min_capacity: usize) {
        if min_capacity < self.buf.len() && size_of::<T>() > 0 {
            // SAFETY: `cap` <- `min_capacity`.
            unsafe { self.reallocate(mc, cmp::max(self.len, min_capacity)) };
        }
    }

    /// Converts the vector into a [`Unique<'gc, [T]>`][Unique GC'd slice] to
    /// allow for further mutation.
    ///
    /// Before doing the conversion, this method discards excess capacity like
    /// [`shrink_to_fit`].
    ///
    /// [Unique GC'd slice]: Unique
    /// [`shrink_to_fit`]: Vec::shrink_to_fit
    pub fn into_unique_slice(mut self, mc: &Mutation<'gc>) -> Unique<'gc, [T]> {
        self.shrink_to_fit(mc);
        // SAFETY: `self.len == self.capacity` means that all elements in
        // the buffer is now initialized.
        unsafe {
            let this = ManuallyDrop::new(self);
            let buf = ptr::read(&this.buf);
            debug_assert_eq!(buf.len(), this.len);

            buf.assume_init()
        }
    }

    /// Converts the vector into a [`Gc<'gc, [T]>`][GC'd slice].
    ///
    /// Before doing the conversion, this method discards excess capacity like
    /// [`shrink_to_fit`].
    ///
    /// [GC'd slice]: Gc
    /// [`shrink_to_fit`]: Vec::shrink_to_fit
    pub fn into_gc_slice(self, mc: &Mutation<'gc>) -> Gc<'gc, [T]> {
        self.into_unique_slice(mc).into()
    }

    /// Appends elements to `self` from other buffer.
    ///
    /// # Safety
    ///
    /// `other` must be a valid slice of length `other.len()`.
    #[inline]
    #[track_caller]
    unsafe fn append_elements(&mut self, mc: &Mutation<'gc>, other: *const [T]) {
        let count = other.len();
        self.reserve(mc, count);
        let len = self.len();
        unsafe { ptr::copy_nonoverlapping(other as *const T, self.as_mut_ptr().add(len), count) };
        self.len += count;
    }

    /// Inserts an element at position `index` within the vector, shifting all
    /// elements after it to the right.
    ///
    /// # Panics
    ///
    /// Panics if `index > len`.
    #[track_caller]
    pub fn insert(&mut self, mc: &Mutation<'gc>, index: usize, element: T) {
        #[cold]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("insertion index (is {index}) should be <= len (is {len})");
        }

        let len = self.len();
        if index > len {
            assert_failed(index, len);
        }

        // space for the new element
        self.reserve(mc, 1);

        unsafe {
            // infallible
            // The spot to put the new value
            {
                let p = self.as_mut_ptr().add(index);
                if index < len {
                    // Shift everything over to make space. (Duplicating the
                    // `index`th element into two consecutive places.)
                    ptr::copy(p, p.add(1), len - index);
                }
                // Write it in, overwriting the first copy of the `index`th
                // element.
                ptr::write(p, element);
            }
            self.set_len(len + 1);
        }
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    pub fn push(&mut self, mc: &Mutation<'gc>, value: T) {
        let len = self.len;
        self.reserve(mc, 1);
        // SAFETY: `self.reserve` ensures that there is enough space.
        unsafe {
            self.as_mut_ptr().add(len).write(value);
            self.set_len(len + 1);
        }
    }

    /// Moves all the elements of `other` into `self`, leaving `other` empty.
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    #[inline]
    #[track_caller]
    pub fn append(&mut self, mc: &Mutation<'gc>, other: &mut Self) {
        unsafe {
            self.append_elements(mc, other.as_slice() as _);
            other.set_len(0);
        }
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with the result of
    /// calling the closure `f`. The return values from `f` will end up
    /// in the `Vec` in the order they have been generated.
    ///
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    /// This method uses a closure to create new values on every push. If
    /// you'd rather [`Clone`] a given value, use [`Vec::resize`]. If you
    /// want to use the [`Default`] trait to generate values, you can
    /// pass [`Default::default`] as the second argument.
    #[track_caller]
    pub fn resize_with<F>(&mut self, mc: &Mutation<'gc>, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let len = self.len();
        if new_len > len {
            self.extend_trusted(mc, core::iter::repeat_with(f).take(new_len - len));
        } else {
            self.truncate(new_len);
        }
    }
}

impl<'gc, T: 'gc> Vec<'gc, T> {
    /// Creates a `Vec<'gc, T>` directly from a pointer, a length, and a
    /// capacity.
    ///
    /// # Safety
    ///
    /// `ptr`, `len`, and `capacity` must be previously returned from
    /// [`Vec::into_raw_parts`].
    ///
    /// [`Vec::into_raw_parts`]: Vec::into_raw_parts
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, capacity: usize) -> Self {
        let () = Self::ASSERT_NO_DROP;
        let ptr = ptr::slice_from_raw_parts_mut(ptr.cast::<MaybeUninit<T>>(), capacity);
        let buf = unsafe { Unique::from_ptr(ptr) };
        Self { buf, len }
    }

    /// Decomposes a `Vec<'gc, T>` into its raw components: `(pointer, length,
    /// capacity)`.
    pub fn into_raw_parts(self) -> (*mut T, usize, usize) {
        let mut this = ManuallyDrop::new(self);
        (this.as_mut_ptr(), this.len, this.capacity())
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Shortens the vector, keeping the first `len` elements and dropping
    /// the rest.
    ///
    /// If `len` is greater or equal to the vector's current length, this has
    /// no effect.
    pub fn truncate(&mut self, len: usize) {
        if self.len > len {
            // SAFETY:
            //
            // * the `len` of the vector is shrunk before calling `collect`, such that no
            //   value will be traced twice in case `collect` were to panic once (if it
            //   panics twice, the program aborts).
            unsafe {
                self.set_len(len);
                let () = Self::ASSERT_NO_DROP;
            }
        }
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// Equivalent to `&s[..]`.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `slice::from_raw_parts` requires pointee is a contiguous, aligned
        // buffer of size `len` containing properly-initialized `T`s. Data must
        // not be mutated for the returned lifetime. Further, `len *
        // size_of::<T>` <= `isize::MAX`, and allocation does not "wrap" through
        // overflowing memory addresses.
        //
        // * Vec API guarantees that self.buf:
        //      * contains only properly-initialized items within 0..len
        //      * is aligned, contiguous, and valid for `len` reads
        //      * obeys size and address-wrapping constraints
        //
        // * We only construct `&mut` references to `self.buf` through `&mut self`
        //   methods; borrow- check ensures that it is not possible to mutably alias
        //   `self.buf` within the returned lifetime.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Extracts a mutable slice of the entire vector.
    ///
    /// Equivalent to `&mut s[..]`.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `slice::from_raw_parts_mut` requires pointee is a contiguous, aligned
        // buffer of size `len` containing properly-initialized `T`s. Data must
        // not be accessed through any other pointer for the returned lifetime.
        // Further, `len * size_of::<T>` <= `ISIZE::MAX` and allocation does not
        // "wrap" through overflowing memory addresses.
        //
        // * Vec API guarantees that self.buf:
        //      * contains only properly-initialized items within 0..len
        //      * is aligned, contiguous, and valid for `len` reads
        //      * obeys size and address-wrapping constraints
        //
        // * We only construct references to `self.buf` through `&self` and `&mut self`
        //   methods; borrow-check ensures that it is not possible to construct a
        //   reference to `self.buf` within the returned lifetime.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Returns a raw pointer to the vector's buffer.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up dangling.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    pub fn as_ptr(&self) -> *const T {
        self.buf.as_ptr().cast::<T>()
    }

    /// Returns a raw mutable pointer to the vector's buffer.
    ///
    /// The caller must ensure that the vector outlives the pointer this
    /// function returns, or else it will end up dangling.
    /// Modifying the vector may cause its buffer to be reallocated,
    /// which would also make any pointers to it invalid.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.as_mut_ptr().cast::<T>()
    }

    /// Forces the length of the vector to `new_len`.
    ///
    /// This is a low-level operation that maintains none of the normal
    /// invariants of the type.
    ///
    /// # Safety
    ///
    /// - `new_len` must be less than or equal to [`capacity()`].
    /// - The elements at `old_len..new_len` must be initialized.
    ///
    /// [`capacity()`]: Vec::capacity
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity());
        self.len = new_len;
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering of the remaining elements, but is
    /// *O*(1). If you need to preserve the element order, use [`remove`]
    /// instead.
    ///
    /// [`remove`]: Vec::remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    pub fn swap_remove(&mut self, index: usize) -> T {
        #[cold]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("swap_remove index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }
        unsafe {
            // We replace self[index] with the last element. Note that if the
            // bounds check above succeeds there must be a last element (which
            // can be self[index] itself).
            let value = ptr::read(self.as_ptr().add(index));
            let base_ptr = self.as_mut_ptr();
            ptr::copy(base_ptr.add(len - 1), base_ptr.add(index), 1);
            self.set_len(len - 1);
            value
        }
    }

    /// Removes and returns the element at position `index` within the vector,
    /// shifting all elements after it to the left.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of *O*(*n*). If you don't need the order of
    /// elements to be preserved, use [`swap_remove`] instead.
    ///
    /// [`swap_remove`]: Vec::swap_remove
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[track_caller]
    pub fn remove(&mut self, index: usize) -> T {
        #[cold]
        #[track_caller]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("removal index (is {index}) should be < len (is {len})");
        }

        let len = self.len();
        if index >= len {
            assert_failed(index, len);
        }
        unsafe {
            // infallible
            let ret;
            {
                // the place we are taking from.
                let ptr = self.as_mut_ptr().add(index);
                // copy it out, unsafely having a copy of the value on
                // the stack and in the vector at the same time.
                ret = ptr::read(ptr);

                // Shift everything down to fill in that spot.
                ptr::copy(ptr.add(1), ptr, len - index - 1);
            }
            self.set_len(len - 1);
            ret
        }
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an
    /// error is returned with the element.
    ///
    /// Unlike [`push`] this method will not reallocate when there's
    /// insufficient capacity. The caller should use [`reserve`] to ensure
    /// that there is enough capacity.
    ///
    /// [`push`]: Vec::push
    /// [`reserve`]: Vec::reserve
    #[inline]
    pub fn push_within_capacity(&mut self, value: T) -> Result<(), T> {
        let len = self.len;
        if len == self.capacity() {
            return Err(value);
        }
        // SAFETY: `self.len` is less than `self.capacity()`, which means there
        // is free space.
        unsafe {
            self.as_mut_ptr().add(len).write(value);
            self.set_len(len + 1);
        }
        Ok(())
    }

    /// Removes the last element from a vector and returns it, or [`None`] if it
    /// is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            // SAFETY: `self.len` is non-zero.
            unsafe {
                self.set_len(self.len - 1);
                core::hint::assert_unchecked(self.len < self.capacity());
                Some(ptr::read(self.as_ptr().add(self.len)))
            }
        }
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns `true`, or [`None`] if the predicate returns false or the vector
    /// is empty (the predicate will not be called in that case).
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        let last = self.last_mut()?;
        if predicate(last) { self.pop() } else { None }
    }

    /// Clears the vector, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity
    /// of the vector.
    #[inline]
    pub fn clear(&mut self) {
        // SAFETY:
        // - `elems` comes directly from `as_mut_slice` and is therefore valid.
        unsafe {
            self.set_len(0);
            let () = Self::ASSERT_NO_DROP;
        }
    }

    /// Returns the number of elements in the vector, also referred to
    /// as its 'length'.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vector contains no elements.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the remaining spare capacity of the vector as a slice of
    /// `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the vector with data (e.g. by
    /// reading from a file) before marking the data as initialized using the
    /// [`set_len`] method.
    ///
    /// [`set_len`]: Vec::set_len
    #[inline]
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        // Note:
        // This method is not implemented in terms of `split_at_spare_mut`,
        // to prevent invalidation of pointers to the buffer.
        unsafe {
            slice::from_raw_parts_mut(
                self.as_mut_ptr().add(self.len) as *mut MaybeUninit<T>,
                self.capacity() - self.len,
            )
        }
    }
}

impl<'gc, T: 'gc + Collect<'gc> + Clone> Vec<'gc, T> {
    /// Constructs a `Vec` from `n` elements.
    pub fn from_elem(mc: &Mutation<'gc>, elem: T, n: usize) -> Self {
        let () = Self::ASSERT_NO_DROP;
        SpecFromElem::from_elem(elem, n, mc)
    }

    /// Clones a `Vec`.
    ///
    /// The signature differs from the [`Clone`] trait from the standard library
    /// since a [`Mutation`] is required to handle the allocation.
    pub fn clone(&self, mc: &Mutation<'gc>) -> Self {
        ConvertVec::to_vec(&**self, mc)
    }

    pub fn clone_from(&mut self, mc: &Mutation<'gc>, src: &Self) {
        ConvertVec::clone_into_vec(&**src, mc, self);
    }

    /// Resizes the `Vec` in-place so that `len` is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `Vec` is extended by the
    /// difference, with each additional slot filled with `value`.
    /// If `new_len` is less than `len`, the `Vec` is simply truncated.
    ///
    /// This method requires `T` to implement [`Clone`],
    /// in order to be able to clone the passed value.
    /// If you need more flexibility (or want to rely on [`Default`] instead of
    /// [`Clone`]), use [`Vec::resize_with`].
    /// If you only need to resize to a smaller size, use [`Vec::truncate`].
    ///
    /// # Panics
    ///
    /// Panics if the new capacity exceeds `isize::MAX` _bytes_.
    #[track_caller]
    pub fn resize(&mut self, mc: &Mutation<'gc>, new_len: usize, value: T) {
        let len = self.len();

        if new_len > len {
            self.extend_with(mc, new_len - len, value)
        } else {
            self.truncate(new_len);
        }
    }

    pub fn extend_from_slice(&mut self, mc: &Mutation<'gc>, other: &[T]) {
        SpecExtend::extend(self, mc, other.iter());
    }
}

impl<'gc, T: 'gc + Collect<'gc>> Vec<'gc, T> {
    /// Extends a `Vec` with the contents of an iterator.
    ///
    /// The signature differs from the [`Extend`] trait from the standard
    /// library since a [`Mutation`] is required to handle the allocation.
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, mc: &Mutation<'gc>, iter: I) {
        SpecExtend::extend(self, mc, iter.into_iter());
    }

    /// Extends a `Vec` with the contents of an iterator via cloning.
    ///
    /// The signature differs from the [`Extend`] trait from the standard
    /// library since a [`Mutation`] is required to handle the allocation.
    pub fn extend_ref<'a, I: IntoIterator<Item = &'a T>>(&mut self, mc: &Mutation<'gc>, iter: I)
    where
        T: Clone + 'a,
    {
        SpecExtend::extend(self, mc, iter.into_iter());
    }

    /// Transforms an iterator into a `Vec`.
    ///
    /// The signature of this function differs from [`Iterator::collect`] from
    /// the standard library since a [`Mutation`] is required to handle the
    /// allocation.
    pub fn collect<I: IntoIterator<Item = T>>(mc: &Mutation<'gc>, iter: I) -> Self {
        let () = Self::ASSERT_NO_DROP;
        SpecFromIter::from_iter(mc, iter.into_iter())
    }

    // leaf method to which various SpecFrom/SpecExtend implementations delegate
    // when they have no further optimizations to apply
    #[track_caller]
    fn extend_desugared<I: Iterator<Item = T>>(&mut self, mc: &Mutation<'gc>, mut iterator: I) {
        // This is the case for a general iterator.
        //
        // This function should be the moral equivalent of:
        //
        //      for item in iterator {
        //          self.push(item);
        //      }
        while let Some(element) = iterator.next() {
            let len = self.len();
            if len == self.capacity() {
                let (lower, _) = iterator.size_hint();
                self.reserve(mc, lower.saturating_add(1));
            }
            unsafe {
                ptr::write(self.as_mut_ptr().add(len), element);
                // Since next() executes user code which can panic we have to bump the length
                // after each step.
                // NB can't overflow since we would have had to alloc the address space
                self.set_len(len + 1);
            }
        }
    }

    // specific extend for `TrustedLen` iterators, called both by the
    // specializations and internal places where resolving specialization makes
    // compilation slower
    #[track_caller]
    fn extend_trusted(&mut self, mc: &Mutation<'gc>, iterator: impl TrustedLen<Item = T>) {
        let (low, high) = iterator.size_hint();
        if let Some(additional) = high {
            debug_assert_eq!(
                low,
                additional,
                "TrustedLen iterator's size hint is not exact: {:?}",
                (low, high)
            );
            self.reserve(mc, additional);
            unsafe {
                let ptr = self.as_mut_ptr();
                let mut local_len = SetLenOnDrop::new(&mut self.len);
                iterator.for_each(move |element| {
                    ptr::write(ptr.add(local_len.current_len()), element);
                    // Since the loop executes user code which can panic we have to update
                    // the length every step to correctly drop what we've written.
                    // NB can't overflow since we would have had to alloc the address space
                    local_len.increment_len(1);
                });
            }
        } else {
            // Per TrustedLen contract a `None` upper bound means that the iterator length
            // truly exceeds usize::MAX, which would eventually lead to a capacity overflow
            // anyway. Since the other branch already panics eagerly (via `reserve()`) we do
            // the same here. This avoids additional codegen for a fallback code path which
            // would eventually panic anyway.
            panic!("capacity overflow");
        }
    }

    #[track_caller]
    /// Extend the vector by `n` clones of value.
    fn extend_with(&mut self, mc: &Mutation<'gc>, n: usize, value: T)
    where
        T: Clone,
    {
        self.reserve(mc, n);

        unsafe {
            let mut ptr = self.as_mut_ptr().add(self.len());
            // Use SetLenOnDrop to work around bug where compiler
            // might not realize the store through `ptr` through self.set_len()
            // don't alias.
            let mut local_len = SetLenOnDrop::new(&mut self.len);

            // Write all elements except the last one
            for _ in 1..n {
                ptr::write(ptr, value.clone());
                ptr = ptr.add(1);
                // Increment the length in every step in case clone() panics
                local_len.increment_len(1);
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                ptr::write(ptr, value);
                local_len.increment_len(1);
            }

            // len set by scope guard
        }
    }
}

impl<'gc, T: 'gc> Deref for Vec<'gc, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'gc, T: 'gc> DerefMut for Vec<'gc, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<'gc, T: 'gc + PartialEq<U>, U: 'gc, const N: usize> PartialEq<[U; N]> for Vec<'gc, T> {
    #[inline]
    fn eq(&self, other: &[U; N]) -> bool {
        PartialEq::eq(&**self, other)
    }
}

impl<'gc, T: 'gc + PartialEq<U>, U: 'gc> PartialEq<[U]> for Vec<'gc, T> {
    #[inline]
    fn eq(&self, other: &[U]) -> bool {
        PartialEq::eq(&**self, other)
    }
}

impl<'gc, T: 'gc + PartialEq<U>, U: 'gc> PartialEq<Vec<'gc, U>> for Vec<'gc, T> {
    #[inline]
    fn eq(&self, other: &Vec<'gc, U>) -> bool {
        PartialEq::eq(&**self, &**other)
    }
}

impl<'gc, T: 'gc + Eq> Eq for Vec<'gc, T> {}

impl<'gc, T: 'gc + PartialOrd> PartialOrd<Vec<'gc, T>> for Vec<'gc, T> {
    #[inline]
    fn partial_cmp(&self, other: &Vec<'gc, T>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(&**self, &**other)
    }
}

impl<'gc, T: 'gc + Ord> Ord for Vec<'gc, T> {
    #[inline]
    fn cmp(&self, other: &Vec<'gc, T>) -> cmp::Ordering {
        Ord::cmp(&**self, &**other)
    }
}

impl<'gc, T: 'gc + Hash> Hash for Vec<'gc, T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        Hash::hash(&**self, state)
    }
}

impl<'gc, T: 'gc, I: SliceIndex<[T]>> Index<I> for Vec<'gc, T> {
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&**self, index)
    }
}

impl<'gc, T: 'gc, I: SliceIndex<[T]>> IndexMut<I> for Vec<'gc, T> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut **self, index)
    }
}

impl<'gc, T: 'gc + fmt::Debug> fmt::Debug for Vec<'gc, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<'gc, T: 'gc> AsRef<[T]> for Vec<'gc, T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<'gc, T: 'gc> AsMut<[T]> for Vec<'gc, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<'gc, T: 'gc> IntoIterator for Vec<'gc, T> {
    type Item = T;
    type IntoIter = IntoIter<'gc, T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::from_vec(self)
    }
}

impl<'a, 'gc, T: 'gc> IntoIterator for &'a Vec<'gc, T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, 'gc, T: 'gc> IntoIterator for &'a mut Vec<'gc, T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<'gc, T> From<(&[T], &Mutation<'gc>)> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    fn from((slice, mc): (&[T], &Mutation<'gc>)) -> Self {
        ConvertVec::to_vec(slice, mc)
    }
}

impl<'gc, T> From<(&Mutation<'gc>, &[T])> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    fn from((mc, slice): (&Mutation<'gc>, &[T])) -> Self {
        ConvertVec::to_vec(slice, mc)
    }
}

impl<'gc, T> From<(&mut [T], &Mutation<'gc>)> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((slice, mc): (&mut [T], &Mutation<'gc>)) -> Self {
        ConvertVec::to_vec(slice, mc)
    }
}

impl<'gc, T> From<(&Mutation<'gc>, &mut [T])> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((mc, slice): (&Mutation<'gc>, &mut [T])) -> Self {
        ConvertVec::to_vec(slice, mc)
    }
}

impl<'gc, T, const N: usize> From<(&[T; N], &Mutation<'gc>)> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((array, mc): (&[T; N], &Mutation<'gc>)) -> Self {
        (array.as_slice(), mc).into()
    }
}

impl<'gc, T, const N: usize> From<(&Mutation<'gc>, &[T; N])> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((mc, array): (&Mutation<'gc>, &[T; N])) -> Self {
        (array.as_slice(), mc).into()
    }
}

impl<'gc, T, const N: usize> From<(&mut [T; N], &Mutation<'gc>)> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((array, mc): (&mut [T; N], &Mutation<'gc>)) -> Self {
        (array.as_slice(), mc).into()
    }
}

impl<'gc, T, const N: usize> From<(&Mutation<'gc>, &mut [T; N])> for Vec<'gc, T>
where
    T: Collect<'gc> + Clone + 'gc,
{
    #[inline]
    fn from((mc, array): (&Mutation<'gc>, &mut [T; N])) -> Self {
        (array.as_slice(), mc).into()
    }
}

impl<'gc, T, const N: usize> From<([T; N], &Mutation<'gc>)> for Vec<'gc, T>
where
    T: Collect<'gc> + 'gc,
{
    #[inline]
    fn from((array, mc): ([T; N], &Mutation<'gc>)) -> Self {
        Self::from_array(mc, array)
    }
}

impl<'gc, T, const N: usize> From<(&Mutation<'gc>, [T; N])> for Vec<'gc, T>
where
    T: Collect<'gc> + 'gc,
{
    #[inline]
    fn from((mc, array): (&Mutation<'gc>, [T; N])) -> Self {
        Self::from_array(mc, array)
    }
}

impl<'gc, T: 'gc> From<Unique<'gc, [T]>> for Vec<'gc, T> {
    fn from(slice: Unique<'gc, [T]>) -> Vec<'gc, T> {
        slice.into_vec()
    }
}

impl<'gc, T: 'gc, const N: usize> TryFrom<Vec<'gc, T>> for [T; N] {
    type Error = Vec<'gc, T>;

    fn try_from(mut vec: Vec<'gc, T>) -> Result<Self, Vec<'gc, T>> {
        if vec.len() != N {
            return Err(vec);
        }

        // SAFETY: `.set_len(0)` is always sound.
        unsafe { vec.set_len(0) };

        // SAFETY: A `Vec`'s pointer is always aligned properly, and
        // the alignment the array needs is the same as the items.
        // We checked earlier that we have sufficient items.
        // The items will not double-drop as the `set_len`
        // tells the `Vec` not to also drop them.
        Ok(unsafe { ptr::read(vec.as_ptr() as *const [T; N]) })
    }
}

#[cfg(test)]
mod tests {
    use std::string::ToString;

    use crate::{Gc, arena::rootless_mutate};

    #[test]
    fn macros() {
        rootless_mutate(|mc| {
            let mut vec = crate::vec![mc];
            vec.push(mc, 1);
            vec.push(mc, 2);
            assert_eq!(vec, [1, 2]);

            let v2 = crate::vec![mc => Gc::new(mc, "Hello".to_string()); 1024];
            assert_eq!(v2.len(), 1024);
            assert_eq!(*v2[50], "Hello");

            let v3 = crate::vec![mc => 3, 4, 5, 6, 7];
            assert_eq!(v3, [3, 4, 5, 6, 7]);

            let v4 = crate::vec![mc => (); 100];
            assert_eq!(v4.capacity(), usize::MAX);
        })
    }
}
