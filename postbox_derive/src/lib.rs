//! # postbox_derive — procedural macros for lattice types
//!
//! This crate provides the **derive macros** for the `postbox`
//! library, enabling boilerplate-free implementations of lattice
//! traits.
//!
//! Currently supported derives:
//!
//! - **`#[derive(JoinSemilatticeDerive)]`**
//!   Automatically implements [`JoinSemilattice`] for a struct by
//!   performing the join componentwise across all fields.
//!
//! - **`#[derive(BoundedJoinSemilatticeDerive)]`**
//!   Implements [`BoundedJoinSemilattice`] by calling `bottom()` on
//!   each field and forming the product lattice's bottom.
//!
//! These derive macros are re-exported through `postbox` when the
//! `derive` feature is enabled:
//!
//! ```ignore
//! use postbox::join_semilattice::{JoinSemilattice, BoundedJoinSemilattice};
//! use postbox::{JoinSemilatticeDerive, BoundedJoinSemilatticeDerive};
//!
//! #[derive(JoinSemilatticeDerive, BoundedJoinSemilatticeDerive)]
//! struct Foo {
//!     a: postbox::crdt::GCounterState<&'static str>,
//!     b: std::collections::HashSet<i32>,
//! }
//! ```
//!
//! This crate is **not** intended to be used directly. You normally
//! depend on it implicitly via:
//!
//! ```toml
//! postbox = { version = "...", features = ["derive"] }
//! ```
//!
//! Internally, the macros generate efficient, order-preserving joins
//! and bottom constructors, following the product-lattice semantics
//! used throughout the library.
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;
use syn::parse_quote;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;

/// Internal helper: ensure we're deriving on a struct with named
/// fields.
fn get_named_fields(
    input: &DeriveInput,
) -> Result<&syn::punctuated::Punctuated<syn::Field, syn::token::Comma>, TokenStream> {
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                let msg = "#[derive(JoinSemilattice / BoundedJoinSemilattice)] currently only supports structs with named fields";
                return Err(syn::Error::new_spanned(&input.ident, msg)
                    .to_compile_error()
                    .into());
            }
        },
        _ => {
            let msg =
                "#[derive(JoinSemilattice / BoundedJoinSemilattice)] is only supported on structs";
            return Err(syn::Error::new_spanned(&input.ident, msg)
                .to_compile_error()
                .into());
        }
    };

    Ok(fields)
}

#[proc_macro_derive(JoinSemilattice)]
pub fn derive_join_semilattice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    // Collect field idents and types.
    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    // Clone generics and extend the where clause with:
    //   <field-ty> : ::postbox::join_semilattice::JoinSemilattice
    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::postbox::join_semilattice::JoinSemilattice));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Implement join as fieldwise join.
    let expanded = quote! {
        impl #impl_generics ::postbox::join_semilattice::JoinSemilattice for #name #ty_generics
        #where_clause
        {
            fn join(&self, other: &Self) -> Self {
                #name {
                    #( #field_idents: ::postbox::join_semilattice::JoinSemilattice::join(&self.#field_idents, &other.#field_idents), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

#[proc_macro_derive(BoundedJoinSemilattice)]
pub fn derive_bounded_join_semilattice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    // Collect field idents and types.
    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    // Clone generics and extend the where clause with:
    //   <field-ty> : ::postbox::join_semilattice::BoundedJoinSemilattice
    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::postbox::join_semilattice::BoundedJoinSemilattice));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Implement bottom as struct-of-bottoms.
    let expanded = quote! {
        impl #impl_generics ::postbox::join_semilattice::BoundedJoinSemilattice for #name #ty_generics
        #where_clause
        {
            fn bottom() -> Self {
                #name {
                    #( #field_idents: ::postbox::join_semilattice::BoundedJoinSemilattice::bottom(), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
