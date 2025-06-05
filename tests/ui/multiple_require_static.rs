use gc_arena::Collect;

#[derive(Collect)]
#[collect(no_drop)]
struct MyStruct {
    #[collect(static)]
    #[collect(static)]
    field: bool
}

fn main() {}
