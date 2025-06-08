use proc_macro2::TokenStream;
use quote::{ToTokens, quote};
use syn::punctuated::Punctuated;

pub(crate) fn derive(s: synstructure::Structure) -> TokenStream {
    fn meta_iter<'a>(
        s: &'a [syn::Attribute],
    ) -> impl Iterator<Item = &'a syn::Attribute> + use<'a> {
        s.iter().filter(|a| a.path().is_ident("ptr_metadata"))
    }

    const ARG_ERROR: &str = "`#[ptr_metadata(...)]` requires one type, \
        optionally `via(expr)` and optionally `bound(...)` in an `unsafe(...)`.";

    fn usage_error(meta: &syn::meta::ParseNestedMeta, msg: &str) -> syn::parse::Error {
        meta.error(format_args!("{msg}. {ARG_ERROR}."))
    }

    let mut tt = TokenStream::new();

    for attr in meta_iter(&s.ast().attrs) {
        let mut unsafety = false;
        let mut ty = None;
        let mut via = None;
        let mut override_bound = None;

        let result = attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("unsafe") {
                if unsafety {
                    return Err(usage_error(&meta, "multiple `unsafe`"));
                }
                unsafety = true;

                return meta.parse_nested_meta(|meta| {
                    if meta.path.is_ident("via") {
                        if via.is_some() {
                            return Err(usage_error(&meta, "multiple `via`"));
                        }
                        let content;
                        syn::parenthesized!(content in meta.input);
                        let e: syn::Expr = content.parse()?;
                        via = Some(e);
                        return Ok(());
                    }

                    if meta.path.is_ident("bound") {
                        if override_bound.is_some() {
                            return Err(usage_error(&meta, "multiple bounds specified"));
                        }

                        let content;
                        syn::parenthesized!(content in meta.input);
                        let lit: Punctuated<syn::WherePredicate, syn::Token![,]> =
                            Punctuated::parse_terminated(&content)?;
                        override_bound = Some(lit);
                        return Ok(());
                    }

                    Err(usage_error(&meta, "unknown option"))
                });
            }

            if ty.is_some() {
                return Err(usage_error(&meta, "multiple types specified"));
            }

            meta.input.parse::<syn::parse::Nothing>()?;
            ty = Some(meta.path);

            Ok(())
        });

        if let Err(e) = result {
            return e.to_compile_error();
        }

        if ty.is_none() {
            return syn::Error::new_spanned(attr, ARG_ERROR).into_compile_error();
        }

        let mut impl_struct = s.clone();

        let mut errors = vec![];
        impl_struct.filter(|t| match meta_iter(&t.ast().attrs).next() {
            Some(attr) => {
                errors.push(syn::Error::new_spanned(
                    attr,
                    "field-level `#[ptr_metadata]` is not supported",
                ));
                false
            }
            None => true,
        });

        if let syn::Data::Enum(..) = impl_struct.ast().data {
            impl_struct.variants().iter().for_each(|v| {
                if let Some(attr) = meta_iter(v.ast().attrs).next() {
                    errors.push(syn::Error::new_spanned(
                        attr,
                        "enum variant-level `#[ptr_metadata]` is not supported",
                    ));
                }
            });
        }

        if !errors.is_empty() {
            return errors.into_iter().map(|e| e.to_compile_error()).collect();
        }

        let ty = ty.unwrap();
        let via = via.unwrap_or_else(|| syn::parse_quote!(self));
        let override_bound = override_bound.map(|b| b.into_token_stream());

        tt.extend(impl_struct.gen_impl(quote! {
            gen unsafe impl<'a> ::gc_arena::ptr::PtrMetadata<'a, #ty> for @Self
                where #override_bound
            {
                fn ptr_metadata(self) -> <#ty as ::core::ptr::Pointee>::Metadata {
                    #via
                }
            }
        }));

        tt.extend(impl_struct.gen_impl(quote! {
            use ::core::ptr::NonNull;

            gen unsafe impl<'a> ::gc_arena::ptr::Metadata<'a, #ty> for @Self
                where #override_bound
            {
                type Ptr = NonNull<#ty>;

                type Ref = &'a #ty;

                type MutRef = &'a mut #ty;

                #[inline]
                fn with_addr(self, addr: NonNull<()>) -> NonNull<#ty> {
                    NonNull::from_raw_parts(
                        addr,
                        ::gc_arena::ptr::PtrMetadata::<'a, #ty>::ptr_metadata(self),
                    )
                }

                #[inline]
                fn addr(ptr: NonNull<#ty>) -> NonNull<()> {
                    ptr.cast()
                }

                #[inline]
                unsafe fn as_ref(ptr: NonNull<#ty>) -> &'a #ty {
                    unsafe { ptr.as_ref() }
                }

                #[inline]
                unsafe fn as_mut(mut ptr: NonNull<#ty>) -> &'a mut #ty {
                    unsafe { ptr.as_mut() }
                }
            }
        }));
    }

    tt
}
