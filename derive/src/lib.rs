use quote::ToTokens;
use syn::{
    parse::{Parse, ParseStream},
    visit_mut::VisitMut,
};
use synstructure::decl_derive;

mod collect;
mod ptr_metadata;

decl_derive! {
    [Collect, attributes(collect)] =>
    /// Derives the `Collect` trait needed to trace a gc type.
    ///
    /// To derive `Collect`, an additional attribute is required on the struct/enum called
    /// `collect`. This has several optional arguments, but the only required argument is the derive
    /// strategy. This can be one of
    ///
    /// - `#[collect(static)]` - Adds a `'static` bound, which allows for a no-op trace
    ///   implementation. This is the ideal choice where possible.
    /// - `#[collect(no_drop)]` - The typical safe tracing derive strategy which only has to add a
    ///   requirement that your struct/enum does not have a custom implementation of `Drop`.
    /// - `#[collect(unsafe_drop)]` - The most versatile tracing derive strategy which allows a
    ///   custom drop implementation. However, this strategy can lead to unsoundness if care is not
    ///   taken (see the above explanation of `Drop` interactions).
    ///
    /// If no strategy is provided, then `#[collect(no_drop)]` is used by default.
    ///
    /// The `collect` attribute also accepts a number of optional configuration settings:
    ///
    /// - `#[collect(bound(<code>))]` - Replaces the default generated `where` clause with the
    ///   given code. The canonical pattern is `#[collect(bound($(T: Trait),* $(,)?))]`. Note that
    ///   this option is ignored for `static` mode since the only bound it produces is `Self: 'static`.
    ///   Also note that providing an explicit bound in this way is safe, and only changes the trait
    ///   bounds used to enable the implementation of `Collect`.
    ///
    /// - `#[collect(gc_lifetime = <lifetime>)]` - the `Collect` trait requires a `'gc` lifetime
    ///   parameter. If there is no lifetime parameter on the type, then `Collect` will be
    ///   implemented for all `'gc` lifetimes. If there is one lifetime on the type, this is assumed
    ///   to be the `'gc` lifetime. In the very unusual case that there are two or more lifetime
    ///   parameters, you must specify *which* lifetime should be used as the `'gc` lifetime.
    ///
    /// Options must be passed to the `collect` attribute together, e.g.,
    /// `#[collect(no_drop, bound())]`.
    ///
    /// The `collect` attribute may also be used on any field of an enum or struct, however the
    /// only allowed usage is to specify the strategy as `static` (no other strategies are
    /// allowed, and no optional settings can be specified). This will add a `'static` bound to the
    /// type of the field (regardless of an explicit `bound` setting) in exchange for not having
    /// to trace into the given field (the ideal choice where possible). Note that if the entire
    /// struct/enum is marked with `static` then this is unnecessary.
    collect::derive
}

decl_derive! {
    [PtrMetadata, attributes(ptr_metadata)] =>
    /// Derives the `PtrMetadata` trait.
    ///
    /// To derive `PtrMetadata`, the type should be able to be converted to a pointer metadata type.
    /// This can be done by specifying the `#[ptr_metadata(<type>, unsafe(...))]` attribute on the type.
    /// The specified type must not have any generics outside of the type being derived.
    /// The available options are:
    ///
    /// - `#[ptr_metadata(unsafe(via(<expr>)))]` - Specifies an expression that can be used to
    ///   convert the type to a pointer metadata type. The expression context contains an implicit `self`
    ///   variable pointing to the value being converted. If `via` is not specified, the default
    ///   expression is `self`; however, only `std`-defined types can be a pointer metadata type, so the
    ///   default is (almost) unreachable.
    /// - `#[ptr_metadata(unsafe(bound(<code>)))]` - Replaces the default generated `where` clause with
    ///   the given code. The canonical pattern is `#[ptr_metadata(bound($(T: Trait),* $(,)?))]`.
    ///
    /// Options must be passed to the `ptr_metadata` attribute together. All the options must be
    /// specified at most once.
    ///
    /// The `ptr_metadata` attribute may be specified at the type level multiple times, once for each
    /// pointer metadata type that the type can be converted to. The order of the attributes doesn't
    /// matter.
    ptr_metadata::derive
}

// Not public API; implementation detail of `gc_arena::Rootable!`.
// Replaces all `'_` lifetimes in a type by the specified named lifetime.
// Syntax: `__unelide_lifetimes!('lt; SomeType)`.
#[doc(hidden)]
#[proc_macro]
pub fn __unelide_lifetimes(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct Input {
        lt: syn::Lifetime,
        ty: syn::Type,
    }

    impl Parse for Input {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let lt: syn::Lifetime = input.parse()?;
            let _: syn::Token!(;) = input.parse()?;
            let ty: syn::Type = input.parse()?;
            Ok(Self { lt, ty })
        }
    }

    struct UnelideLifetimes(syn::Lifetime);

    impl VisitMut for UnelideLifetimes {
        fn visit_lifetime_mut(&mut self, i: &mut syn::Lifetime) {
            if i.ident == "_" {
                *i = self.0.clone();
            }
        }
    }

    let mut input = syn::parse_macro_input!(input as Input);
    UnelideLifetimes(input.lt).visit_type_mut(&mut input.ty);
    input.ty.to_token_stream().into()
}
