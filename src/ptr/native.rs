use core::ptr::Pointee;

#[repr(transparent)]
pub struct Unsized<Dyn: ?Sized>(pub(crate) <Dyn as Pointee>::Metadata);

impl<Dyn: ?Sized> PartialEq for Unsized<Dyn> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<Dyn: ?Sized> Clone for Unsized<Dyn> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Dyn: ?Sized> Copy for Unsized<Dyn> {}
