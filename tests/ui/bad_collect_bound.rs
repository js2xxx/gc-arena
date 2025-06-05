use gc_arena::Collect;

struct NotCollect;

#[derive(Collect)]
struct MyStruct {
    field: NotCollect
}

fn main() {}
