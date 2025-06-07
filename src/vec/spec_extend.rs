use core::{iter::TrustedLen, slice};

use super::{IntoIter, Vec};
use crate::{Collect, Mutation};

pub(super) trait SpecExtend<'gc, T, I> {
    #[track_caller]
    fn extend(&mut self, mc: &Mutation<'gc>, iter: I);
}

impl<'gc, T, I> SpecExtend<'gc, T, I> for Vec<'gc, T>
where
    I: Iterator<Item = T>,
    T: Collect<'gc>,
{
    #[track_caller]
    default fn extend(&mut self, mc: &Mutation<'gc>, iter: I) {
        self.extend_desugared(mc, iter)
    }
}

impl<'gc, T, I> SpecExtend<'gc, T, I> for Vec<'gc, T>
where
    I: TrustedLen<Item = T>,
    T: Collect<'gc>,
{
    #[track_caller]
    default fn extend(&mut self, mc: &Mutation<'gc>, iterator: I) {
        self.extend_trusted(mc, iterator)
    }
}

/// The differed `'gc` and `'gc1` don't matter since:
///
/// - The buffer in `IntoIter<'gc1, T>` is not touched.
/// - `T: Collect<'gc>` means that elements of type `T` can be collected in this
///   arena:
///   - If `T` contains a GC'd pointer, then the GC'd pointer only implements
///     `Collect<'gc>` because of invariance, so it can only be collected in
///     this arena;
///   - If `T` doesn't contain any non-GC'd pointer, than it can be collected in
///     any arena.
impl<'gc, 'gc1, T: Collect<'gc>> SpecExtend<'gc, T, IntoIter<'gc1, T>> for Vec<'gc, T> {
    #[track_caller]
    fn extend(&mut self, mc: &Mutation<'gc>, mut iterator: IntoIter<'gc1, T>) {
        unsafe { self.append_elements(mc, iterator.as_slice() as _) };
        iterator.forget_remaining();
    }
}

impl<'gc, 'a, T, I> SpecExtend<'gc, &'a T, I> for Vec<'gc, T>
where
    I: Iterator<Item = &'a T>,
    T: Clone + Collect<'gc>,
{
    #[track_caller]
    default fn extend(&mut self, mc: &Mutation<'gc>, iterator: I) {
        self.extend(mc, iterator.cloned())
    }
}

impl<'gc, 'a, T: 'a> SpecExtend<'gc, &'a T, slice::Iter<'a, T>> for Vec<'gc, T>
where
    T: Copy + Collect<'gc>,
{
    #[track_caller]
    fn extend(&mut self, mc: &Mutation<'gc>, iterator: slice::Iter<'a, T>) {
        let slice = iterator.as_slice();
        unsafe { self.append_elements(mc, slice) };
    }
}
