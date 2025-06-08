use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, quote, quote_spanned};
use syn::{punctuated::Punctuated, spanned::Spanned};
use synstructure::AddBounds;

pub(crate) fn derive(s: synstructure::Structure) -> TokenStream {
    fn find_collect_meta(attrs: &[syn::Attribute]) -> syn::Result<Option<&syn::Attribute>> {
        let mut found = None;
        for attr in attrs {
            if attr.path().is_ident("collect") && found.replace(attr).is_some() {
                return Err(syn::parse::Error::new_spanned(
                    attr.path(),
                    "Cannot specify multiple `#[collect]` attributes! Consider merging them.",
                ));
            }
        }

        Ok(found)
    }

    // Deriving `Collect` must be done with care, because an implementation of
    // `Drop` is not necessarily safe for `Collect` types. This derive macro has
    // three available modes to ensure that this is safe:
    //   1) Require that the type be 'static with `#[collect(static)]`.
    //   2) Prohibit a `Drop` impl on the type with `#[collect(no_drop)]`
    //   3) Allow a custom `Drop` impl that might be unsafe with
    //      `#[collect(unsafe_drop)]`. Such `Drop` impls must *not* access garbage
    //      collected pointers during `Drop::drop`.
    #[derive(PartialEq)]
    enum Mode {
        RequireStatic,
        NoDrop,
        UnsafeDrop,
    }

    let mut mode = None;
    let mut is_ref = false;
    let mut override_bound = None;
    let mut gc_lifetime = None;

    fn usage_error(meta: &syn::meta::ParseNestedMeta, msg: &str) -> syn::parse::Error {
        meta.error(format_args!(
            "{msg}. `#[collect(...)]` requires one mode \
            (`static`, `no_drop` (`ref`?), or `unsafe_drop` (`ref`?)) and optionally `bound(...)`."
        ))
    }

    let result = match find_collect_meta(&s.ast().attrs) {
        Ok(Some(attr)) => attr.parse_nested_meta(|meta| {
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

            if meta.path.is_ident("gc_lifetime") {
                if gc_lifetime.is_some() {
                    return Err(usage_error(&meta, "multiple `'gc` lifetimes specified"));
                }

                let lit: syn::Lifetime = meta.value()?.parse()?;
                gc_lifetime = Some(lit);
                return Ok(());
            }

            meta.input.parse::<syn::parse::Nothing>()?;

            if mode.is_some() {
                return Err(usage_error(&meta, "multiple modes specified"));
            } else if meta.path.is_ident("static") {
                mode = Some(Mode::RequireStatic);
            } else if meta.path.is_ident("no_drop") {
                mode = Some(Mode::NoDrop);
            } else if meta.path.is_ident("unsafe_drop") {
                mode = Some(Mode::UnsafeDrop);
            } else if meta.path.is_ident("ref") {
                is_ref = true;
            } else {
                return Err(usage_error(&meta, "unknown option"));
            }
            Ok(())
        }),
        Ok(None) => Ok(()),
        Err(err) => Err(err),
    };

    if let Err(err) = result {
        return err.to_compile_error();
    }

    let mode = mode.unwrap_or(Mode::NoDrop);

    if is_ref && mode == Mode::RequireStatic {
        return quote_spanned! {Span::call_site() =>
            compile_error!("`#[collect(static)]` cannot be used with `#[collect(ref)]`");
        };
    }

    let where_clause: TokenStream = if mode == Mode::RequireStatic {
        quote!(where Self: 'static)
    } else {
        quote!(where #override_bound)
    };

    let mut errors = vec![];

    let collect_impl = if mode == Mode::RequireStatic {
        let mut impl_struct = s.clone();
        impl_struct.add_bounds(AddBounds::None);
        impl_struct.gen_impl(quote! {
            gen unsafe impl<'gc> ::gc_arena::Collect<'gc> for @Self #where_clause {
                const NEEDS_TRACE: bool = false;
            }

            gen unsafe impl<'gc> ::gc_arena::collect::CollectRef<'gc> for @Self #where_clause {
                const NEEDS_TRACE: bool = false;
            }
        })
    } else {
        let trait_ = if is_ref {
            quote!(::gc_arena::collect::CollectRef)
        } else {
            quote!(::gc_arena::Collect)
        };

        let trace_func = if is_ref {
            quote!(trace_ref)
        } else {
            quote!(trace)
        };

        let self_arg = if is_ref {
            quote!(&self)
        } else {
            quote!(&mut self)
        };

        let mut impl_struct = s.clone();

        let mut needs_trace_expr = TokenStream::new();
        quote!(false).to_tokens(&mut needs_trace_expr);

        let mut static_bindings = vec![];

        // Ignore all bindings that have `#[collect(static)]` For each binding with
        // `#[collect(static)]`, we push a bound of the form `FieldType: 'static` to
        // `static_bindings`, which will be added to the genererated `Collect` impl. The
        // presence of the bound guarantees that the field cannot hold any `Gc`
        // pointers, so it's safe to ignore that field in `needs_trace` and
        // `trace`
        impl_struct.filter(|b| match find_collect_meta(&b.ast().attrs) {
            Ok(Some(attr)) => {
                let mut static_binding = false;
                let result = attr.parse_nested_meta(|meta| {
                    if meta.input.is_empty() && meta.path.is_ident("static") {
                        static_binding = true;
                        static_bindings.push(b.ast().ty.clone());
                        Ok(())
                    } else {
                        Err(meta.error("Only `#[collect(static)]` is supported on a field"))
                    }
                });
                errors.extend(result.err());
                !static_binding
            }
            Ok(None) => true,
            Err(err) => {
                errors.push(err);
                true
            }
        });

        for static_binding in static_bindings {
            impl_struct.add_where_predicate(syn::parse_quote! { #static_binding: 'static });
        }

        // `#[collect(static)]` only makes sense on fields, not enum variants. Emit an
        // error if it is used in the wrong place
        if let syn::Data::Enum(..) = impl_struct.ast().data {
            for v in impl_struct.variants() {
                for attr in v.ast().attrs {
                    if attr.path().is_ident("collect") {
                        errors.push(syn::parse::Error::new_spanned(
                            attr.path(),
                            "`#[collect]` is not suppported on enum variants",
                        ));
                    }
                }
            }
        }

        // We've already called `impl_struct.filter`, so we we won't try to include
        // `NEEDS_TRACE` for the types of fields that have `#[collect(static)]`
        for v in impl_struct.variants() {
            for b in v.bindings() {
                let ty = &b.ast().ty;
                // Resolving the span at the call site makes rustc emit a 'the error originates
                // a derive macro note' We only use this span on tokens that need to resolve to
                // items (e.g. `gc_arena::Collect`), so this won't cause any hygiene issues
                let call_span = b.ast().span().resolved_at(Span::call_site());
                quote_spanned!(call_span=>
                    || <#ty as #trait_>::NEEDS_TRACE
                )
                .to_tokens(&mut needs_trace_expr);
            }
        }

        if !is_ref {
            impl_struct.bind_with(|_| synstructure::BindStyle::RefMut);
        }

        // Likewise, this will skip any fields that have `#[collect(static)]`
        let trace_body = impl_struct.each(|bi| {
            // See the above handling of `NEEDS_TRACE` for an explanation of this
            let call_span = bi.ast().span().resolved_at(Span::call_site());
            quote_spanned!(call_span=>
                {
                    // Use a temporary variable to ensure that all tokens in the call to
                    // `gc_arena::Collect::trace` have the same hygiene information. If we used
                    // #bi directly, then we would have a mix of hygiene contexts, which would
                    // cause rustc to produce sub-optimal error messagse due to its inability to
                    // merge the spans. This is purely for diagnostic purposes, and has no effect
                    // on correctness
                    let bi = #bi;
                    cc.#trace_func(bi);
                }
            )
        });

        // If we have no configured `'gc` lifetime and the type has a *single* generic
        // lifetime, use that one.
        if gc_lifetime.is_none() {
            let mut all_lifetimes =
                impl_struct
                    .ast()
                    .generics
                    .params
                    .iter()
                    .filter_map(|p| match p {
                        syn::GenericParam::Lifetime(lt) => Some(lt),
                        _ => None,
                    });

            if let Some(lt) = all_lifetimes.next() {
                if all_lifetimes.next().is_none() {
                    gc_lifetime = Some(lt.lifetime.clone());
                } else {
                    panic!(
                        "deriving `Collect` on a type with multiple lifetime parameters requires a `#[collect(gc_lifetime = ...)]` attribute"
                    );
                }
            }
        };

        if override_bound.is_some() {
            impl_struct.add_bounds(AddBounds::None);
        } else {
            impl_struct.add_bounds(AddBounds::Generics);
        };

        let mut tt = if let Some(gc_lifetime) = &gc_lifetime {
            impl_struct.gen_impl(quote! {
                gen unsafe impl #trait_<#gc_lifetime> for @Self #where_clause {
                    const NEEDS_TRACE: bool = #needs_trace_expr;

                    #[inline]
                    fn #trace_func<Trace: ::gc_arena::collect::Trace<#gc_lifetime>>(
                        #self_arg,
                        cc: &mut Trace
                    ) {
                        match *self { #trace_body }
                    }
                }
            })
        } else {
            impl_struct.gen_impl(quote! {
                gen unsafe impl<'gc> #trait_<'gc> for @Self #where_clause {
                    const NEEDS_TRACE: bool = #needs_trace_expr;

                    #[inline]
                    fn #trace_func<Trace: ::gc_arena::collect::Trace<'gc>>(
                        #self_arg,
                        cc: &mut Trace
                    ) {
                        match *self { #trace_body }
                    }
                }
            })
        };

        if is_ref {
            tt.extend(if let Some(gc_lifetime) = gc_lifetime {
                impl_struct.gen_impl(quote! {
                    gen unsafe impl ::gc_arena::Collect<#gc_lifetime> for @Self #where_clause {
                        const NEEDS_TRACE = <Self as #trait_<#gc_lifetime>>::NEEDS_TRACE;

                        #[inline]
                        fn trace<Trace: ::gc_arena::collect::Trace<#gc_lifetime>>(
                            &mut self,
                            cc: &mut Trace
                        ) {
                            <Self as #trait_<#gc_lifetime>>::#trace_func(self, cc);
                        }
                    }
                })
            } else {
                impl_struct.gen_impl(quote! {
                    gen unsafe impl<'gc> ::gc_arena::Collect<'gc> for @Self #where_clause {
                        const NEEDS_TRACE = <Self as #trait_<'gc>>::NEEDS_TRACE;

                        #[inline]
                        fn trace<Trace: ::gc_arena::collect::Trace<'gc>>(
                            &mut self,
                            cc: &mut Trace
                        ) {
                            <Self as #trait_<'gc>>::#trace_func(self, cc);
                        }
                    }
                })
            });
        }

        tt
    };

    let drop_impl = if matches!(mode, Mode::NoDrop) {
        let mut drop_struct = s.clone();
        drop_struct.add_bounds(AddBounds::None).gen_impl(quote! {
            gen impl ::gc_arena::__MustNotImplDrop for @Self {}
        })
    } else {
        quote!()
    };

    let errors = errors.into_iter().map(|e| e.to_compile_error());
    quote! {
        #collect_impl
        #drop_impl
        #(#errors)*
    }
}
