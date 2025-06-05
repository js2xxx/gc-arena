use gc_arena::Collect;

#[derive(Collect)]
struct Foo {
}

impl Drop for Foo {
    fn drop(&mut self) {}
}

fn main() {}
