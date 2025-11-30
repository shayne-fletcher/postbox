#![deny(missing_docs)]
//! # algebra-core-derive — procedural macros for algebraic traits
//!
//! This crate provides **derive macros** for the `algebra-core` library,
//! enabling boilerplate-free implementations of algebraic traits for
//! product types (structs with named fields).
//!
//! ## Supported derives
//!
//! ### Semigroup hierarchy
//! - **`#[derive(Semigroup)]`** — implements `combine` by combining each field
//! - **`#[derive(Monoid)]`** — implements `empty()` by calling `empty()` on each field
//! - **`#[derive(CommutativeMonoid)]`** — marker trait requiring `Monoid`
//! - **`#[derive(Group)]`** — implements `inverse` by inverting each field
//! - **`#[derive(AbelianGroup)]`** — marker trait requiring `Group + CommutativeMonoid`
//!
//! ### Join-semilattice hierarchy
//! - **`#[derive(JoinSemilattice)]`** — implements `join` by joining each field
//! - **`#[derive(BoundedJoinSemilattice)]`** — implements `bottom()` by calling `bottom()` on each field
//!
//! ## Usage
//!
//! These macros are re-exported through `algebra-core` when the `derive` feature is enabled:
//!
//! ```ignore
//! use algebra_core::{Semigroup, Monoid, JoinSemilattice, BoundedJoinSemilattice};
//!
//! #[derive(Clone, PartialEq, Eq, Debug)]
//! #[derive(Semigroup, Monoid, JoinSemilattice, BoundedJoinSemilattice)]
//! struct MyLattice {
//!     counter: algebra_core::Max<i32>,
//!     tags: std::collections::HashSet<String>,
//! }
//! ```
//!
//! Each derive macro generates efficient componentwise implementations
//! following standard product algebra semantics.
use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;
use syn::parse_quote;
use syn::Data;
use syn::DeriveInput;
use syn::Fields;

/// Internal helper: ensure we're deriving on a struct with named fields.
fn get_named_fields(
    input: &DeriveInput,
) -> Result<&syn::punctuated::Punctuated<syn::Field, syn::token::Comma>, TokenStream> {
    let fields = match &input.data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(named) => &named.named,
            _ => {
                let msg = "derive macros currently only support structs with named fields";
                return Err(syn::Error::new_spanned(&input.ident, msg)
                    .to_compile_error()
                    .into());
            }
        },
        _ => {
            let msg = "derive macros are only supported on structs";
            return Err(syn::Error::new_spanned(&input.ident, msg)
                .to_compile_error()
                .into());
        }
    };

    Ok(fields)
}

/// Derive macro for [`Semigroup`](https://docs.rs/algebra-core/latest/algebra_core/trait.Semigroup.html).
///
/// Implements `Semigroup` for a struct by combining each field componentwise.
///
/// # Example
///
/// ```ignore
/// #[derive(Semigroup)]
/// struct Foo {
///     a: i32,  // i32: Semigroup (addition)
///     b: String,  // String: Semigroup (concatenation)
/// }
/// ```
#[proc_macro_derive(Semigroup)]
pub fn derive_semigroup(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::Semigroup));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::Semigroup for #name #ty_generics
        #where_clause
        {
            fn combine(&self, other: &Self) -> Self {
                #name {
                    #( #field_idents: ::algebra_core::Semigroup::combine(&self.#field_idents, &other.#field_idents), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`Monoid`](https://docs.rs/algebra-core/latest/algebra_core/trait.Monoid.html).
///
/// Implements `Monoid` for a struct by constructing `empty()` from each field's empty.
///
/// # Example
///
/// ```ignore
/// #[derive(Semigroup, Monoid)]
/// struct Foo {
///     a: i32,  // i32: Monoid (empty = 0)
///     b: Vec<String>,  // Vec: Monoid (empty = [])
/// }
/// ```
#[proc_macro_derive(Monoid)]
pub fn derive_monoid(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::Monoid));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::Monoid for #name #ty_generics
        #where_clause
        {
            fn empty() -> Self {
                #name {
                    #( #field_idents: ::algebra_core::Monoid::empty(), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`CommutativeMonoid`](https://docs.rs/algebra-core/latest/algebra_core/trait.CommutativeMonoid.html).
///
/// Marker trait indicating that `combine` is commutative.
/// Requires each field to implement `CommutativeMonoid`.
///
/// # Example
///
/// ```ignore
/// #[derive(Semigroup, Monoid, CommutativeMonoid)]
/// struct Foo {
///     a: i32,  // addition is commutative
///     b: std::collections::HashSet<String>,  // union is commutative
/// }
/// ```
#[proc_macro_derive(CommutativeMonoid)]
pub fn derive_commutative_monoid(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::CommutativeMonoid));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::CommutativeMonoid for #name #ty_generics
        #where_clause
        {}
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`Group`](https://docs.rs/algebra-core/latest/algebra_core/trait.Group.html).
///
/// Implements `Group` for a struct by inverting each field componentwise.
///
/// # Example
///
/// ```ignore
/// #[derive(Semigroup, Monoid, Group)]
/// struct Foo {
///     a: i32,  // inverse = negation
///     b: MyGroup,  // custom group
/// }
/// ```
#[proc_macro_derive(Group)]
pub fn derive_group(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::Group));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::Group for #name #ty_generics
        #where_clause
        {
            fn inverse(&self) -> Self {
                #name {
                    #( #field_idents: ::algebra_core::Group::inverse(&self.#field_idents), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`AbelianGroup`](https://docs.rs/algebra-core/latest/algebra_core/trait.AbelianGroup.html).
///
/// Marker trait indicating a commutative group.
/// Requires each field to implement `AbelianGroup`.
///
/// # Example
///
/// ```ignore
/// #[derive(Semigroup, Monoid, CommutativeMonoid, Group, AbelianGroup)]
/// struct Foo {
///     a: i32,  // (Z, +) is abelian
///     b: MyAbelianGroup,
/// }
/// ```
#[proc_macro_derive(AbelianGroup)]
pub fn derive_abelian_group(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::AbelianGroup));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::AbelianGroup for #name #ty_generics
        #where_clause
        {}
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`JoinSemilattice`](https://docs.rs/algebra-core/latest/algebra_core/trait.JoinSemilattice.html).
///
/// Implements `JoinSemilattice` for a struct by joining each field componentwise.
///
/// # Example
///
/// ```ignore
/// #[derive(JoinSemilattice)]
/// struct Foo {
///     counter: Max<i32>,  // join = max
///     tags: HashSet<String>,  // join = union
/// }
/// ```
#[proc_macro_derive(JoinSemilattice)]
pub fn derive_join_semilattice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::JoinSemilattice));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::JoinSemilattice for #name #ty_generics
        #where_clause
        {
            fn join(&self, other: &Self) -> Self {
                #name {
                    #( #field_idents: ::algebra_core::JoinSemilattice::join(&self.#field_idents, &other.#field_idents), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}

/// Derive macro for [`BoundedJoinSemilattice`](https://docs.rs/algebra-core/latest/algebra_core/trait.BoundedJoinSemilattice.html).
///
/// Implements `BoundedJoinSemilattice` for a struct by constructing `bottom()`
/// from each field's bottom element.
///
/// # Example
///
/// ```ignore
/// #[derive(JoinSemilattice, BoundedJoinSemilattice)]
/// struct Foo {
///     counter: Max<i32>,  // bottom = i32::MIN
///     tags: HashSet<String>,  // bottom = ∅
/// }
/// ```
#[proc_macro_derive(BoundedJoinSemilattice)]
pub fn derive_bounded_join_semilattice(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let fields = match get_named_fields(&input) {
        Ok(f) => f,
        Err(ts) => return ts,
    };

    let field_idents: Vec<_> = fields
        .iter()
        .map(|f| f.ident.as_ref().expect("named fields only"))
        .collect();
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let mut generics = input.generics.clone();
    {
        let where_clause = generics.make_where_clause();
        for ty in &field_types {
            where_clause
                .predicates
                .push(parse_quote!(#ty: ::algebra_core::BoundedJoinSemilattice));
        }
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics ::algebra_core::BoundedJoinSemilattice for #name #ty_generics
        #where_clause
        {
            fn bottom() -> Self {
                #name {
                    #( #field_idents: ::algebra_core::BoundedJoinSemilattice::bottom(), )*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
