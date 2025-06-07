#![feature(ptr_metadata)]

use std::{
    alloc::{Layout, LayoutError},
    mem::MaybeUninit,
    ptr::NonNull,
};

use gc_arena::{Collect, arena::rootless_mutate, collect::Trace, gc::Unique, ptr::*};

#[test]
fn custom_layout() {
    #[derive(Debug, Clone, Copy, PartialEq, PtrMetadata)]
    #[ptr_metadata(Memory, unsafe(via(self.size as usize)))]
    #[ptr_metadata(AnotherMemory, unsafe(via(self.size as usize)))]
    struct CompactLayout {
        size: u32,
        align_bit: u8,
    }

    #[derive(Collect)]
    struct Memory([MaybeUninit<u8>]);

    #[derive(Collect)]
    struct AnotherMemory([u8]);

    unsafe impl Uninit for Memory {
        type Init = Self;
    }

    unsafe impl<'a> MetaLayout<'a, Memory> for CompactLayout {
        fn layout(self) -> Result<Layout, LayoutError> {
            Layout::from_size_align(self.size as usize, 1 << self.align_bit)
        }

        unsafe fn drop_in_place(_: NonNull<Memory>) {}
    }

    unsafe impl<'gc, 'a> MetaCollect<'gc, 'a, Memory> for CompactLayout {
        fn needs_trace(self) -> bool {
            Memory::NEEDS_TRACE
        }

        fn trace<C: Trace<'gc>>(_: &'a Memory, _: &mut C) {}
    }

    rootless_mutate(|mc| {
        // Layout { size: 16, align: (1 << 10) == 1024 }
        let layout = CompactLayout { size: 16, align_bit: 10 };

        let unique = Unique::with_metadata::<true>(mc, layout);
        assert!(unique.0.iter().all(|&b| unsafe { b.assume_init() == 0 }));
        let addr = unique.0.as_ptr().addr();
        assert_eq!(addr & ((1 << layout.align_bit) - 1), 0);
    })
}
