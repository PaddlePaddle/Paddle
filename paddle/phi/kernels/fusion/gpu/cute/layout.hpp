/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <cute/config.hpp>

#include <cute/int_tuple.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/stride.hpp>
#include <cute/underscore.hpp>

namespace cute {

// Aliases

template <class... Shapes>
using Shape = IntTuple<Shapes...>;

template <class... Strides>
using Stride = IntTuple<Strides...>;

template <class... Strides>
using Step = IntTuple<Strides...>;

template <class... Coords>
using Coord = IntTuple<Coords...>;

template <class... Ts>
CUTE_HOST_DEVICE constexpr Shape<Ts...> make_shape(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr Stride<Ts...> make_stride(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr Step<Ts...> make_step(Ts const&... t) {
  return {t...};
}
template <class... Ts>
CUTE_HOST_DEVICE constexpr Coord<Ts...> make_coord(Ts const&... t) {
  return {t...};
}

template <class LogicalShape, class LogicalStride = ColMajor<LogicalShape>>
struct Layout : private cute::tuple<LogicalShape,
                                    LogicalStride>  // EBO for static layouts
{
  // Avoid bad CTAD:
  //   Layout smem = GMMA::Layout_MN_SW128_Atom<T>;
  // Should fail because smem is a ComposedLayout (SwizzleLayout) and not a
  // Layout
  static_assert(is_integral<LogicalShape>::value ||
                is_tuple<LogicalShape>::value);

  // Expensive in compilation time...
  // static_assert(is_congruent<LogicalShape, LogicalStride>::value,
  //              "Shape and Stride must have the same hierarchical structure");
  // static_assert(is_integral<LogicalShape>::value ||
  // is_tuple<LogicalShape>::value);

  // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
  CUTE_HOST_DEVICE constexpr Layout(LogicalShape const& logical_shape = {},
                                    LogicalStride const& logical_stride = {})
      : cute::tuple<LogicalShape, LogicalStride>(logical_shape,
                                                 logical_stride) {}

  //
  // Accessors
  //

  static constexpr int rank = rank_v<LogicalShape>;

  CUTE_HOST_DEVICE constexpr decltype(auto) layout() { return *this; }

  CUTE_HOST_DEVICE constexpr decltype(auto) layout() const { return *this; }

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) shape() {
    return get<0, I...>(
        static_cast<cute::tuple<LogicalShape, LogicalStride>&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) shape() const {
    return get<0, I...>(
        static_cast<cute::tuple<LogicalShape, LogicalStride> const&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) stride() {
    return get<1, I...>(
        static_cast<cute::tuple<LogicalShape, LogicalStride>&>(*this));
  }

  template <int... I>
  CUTE_HOST_DEVICE constexpr decltype(auto) stride() const {
    return get<1, I...>(
        static_cast<cute::tuple<LogicalShape, LogicalStride> const&>(*this));
  }

  //
  // Mappings
  //

  // Map a logical coordinate to a linear index (Coord has no Underscore slice
  // operators) OR Slice the layout and return the sublayout (Coord has an
  // Underscore slice op)
  template <class Coord>
  CUTE_HOST_DEVICE constexpr auto operator()(Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      return slice(coord, *this);
    } else {
      return crd2idx(coord, shape(), stride());
    }

    CUTE_GCC_UNREACHABLE;
  }

  // Convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr auto operator()(Coord0 const& c0,
                                             Coord1 const& c1,
                                             Coords const&... cs) const {
    return operator()(make_coord(c0, c1, cs...));
  }

  // Map a linear index to a hier ND logical coordinate
  // NOTE: Dangerous and error-prone
  template <class Int>
  CUTE_HOST_DEVICE constexpr auto operator[](Int const& linear_idx) const {
    static_assert(is_integral<Int>::value);
    return get_hier_coord(linear_idx);
  }

  //
  // Compose
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr auto compose(OtherLayout const& other) const {
    return composition(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto compose(Layouts const&... layouts) const {
    return composition(*this, make_tile(layouts...));
  }

  template <class OtherShape>
  CUTE_HOST_DEVICE constexpr auto with_shape(OtherShape const& shape) const {
    return composition(*this, make_layout(shape));
  }

  template <class... Shapes>
  CUTE_HOST_DEVICE constexpr auto with_shape(Shapes const&... shapes) const {
    return composition(*this, make_layout(make_shape(shapes...)));
  }

  //
  // Tile
  //

  template <class OtherLayout>
  CUTE_HOST_DEVICE constexpr auto tile(OtherLayout const& other) const {
    return tiled_divide(*this, other);
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto tile(Layouts const&... layouts) const {
    return tiled_divide(*this, make_tile(layouts...));
  }

  //
  // Utility
  //

  //
  // Index to Coordinate
  //

  // NOTE: Only valid for compact layouts

  // Return the (hierarchical) ND logical coordinate corresponding to the linear
  // index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post congruent(@a result, shape())
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_hier_coord(IInt const& idx) const {
    return cute::idx2crd(idx, shape(), stride());
  }

  // Return the (flat) ND logical coordinate corresponding to the linear index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post rank(@a result) == rank(shape()) && depth(@a result) == 1
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_flat_coord(IInt const& idx) const {
    return cute::crd2crd(
        this->get_hier_coord(idx), shape(), repeat<rank>(Int<1>{}));
  }

  // Return the generalized column-major 1D logical coordinate corresponding to
  // the linear index
  // @post crd2idx(@a result, shape(), stride()) == idx
  // @post is_integral<decltype(@a result)>::value
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_1d_coord(IInt const& idx) const {
    return cute::crd2idx(this->get_hier_coord(idx), shape());
  }

  //
  // Coordinate to Coordinate
  //

#if 0
  // Return the (hierarchical) ND logical coordinate corresponding to the linear index
  // @post congruent(@a result, shape())
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_hier_coord(Coord const& crd) const {
    return cute::crd2crd(crd, shape(), shape());
  }

  // Return the (flat) ND logical coordinate corresponding to the linear index
  // @post rank(@a result) == rank(shape()) && depth(@a result) == 1
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_flat_coord(Coord const& crd) const {
    return cute::crd2crd(crd, shape(), product_each(shape()));
  }

  // Return the generalized column-major 1D logical coordinate corresponding to the linear index
  // @post is_integral<decltype(@a result)>::value
  template <class Coord>
  CUTE_HOST_DEVICE constexpr
  auto
  crd_2_1d_coord(Coord const& crd) const {
    //return cute::crd2crd(crd, shape(), product(shape()));
    return cute::crd2idx(crd, shape());
  }
#endif
};

template <class Layout>
struct is_layout : false_type {};
template <class Shape, class Stride>
struct is_layout<Layout<Shape, Stride>> : true_type {};

template <
    class Shape,
    class Stride,
    __CUTE_REQUIRES((is_tuple<Shape>::value || is_integral<Shape>::value) &&
                    (is_tuple<Stride>::value || is_integral<Stride>::value))>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape,
                                            Stride const& stride) {
  return Layout<Shape, Stride>(shape, stride);
}

template <class Shape,
          __CUTE_REQUIRES(is_tuple<Shape>::value || is_integral<Shape>::value)>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape) {
  return make_layout(shape, compact_col_major(shape));
}

// Construct a layout from multiple layouts by
//   concatenating each layout as an independent mode
template <class... Shapes, class... Strides>
CUTE_HOST_DEVICE constexpr auto make_layout(
    Layout<Shapes, Strides> const&... layouts) {
  return make_layout(make_shape(layouts.shape()...),
                     make_stride(layouts.stride()...));
}

//
// Convenience tags for common layouts
//

template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape, GenColMajor) {
  return make_layout(shape, compact_col_major(shape));
}

template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_layout(Shape const& shape, GenRowMajor) {
  return make_layout(shape, compact_row_major(shape));
}

// Follow the same ordering induced by the strides, but make the layout compact
template <class Shape, class Order>
CUTE_HOST_DEVICE constexpr auto make_ordered_layout(Shape const& shape,
                                                    Order const& order) {
  static_assert(is_static<Shape>::value && is_static<Order>::value);
  return make_layout(shape, compact_order(shape, order));
}

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_ordered_layout(
    Layout<Shape, Stride> const& layout) {
  return make_ordered_layout(layout.shape(), layout.stride());
}

// Make a layout of the same shape that is either ordered or colmajor depending
// on staticness
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_layout_like(
    Layout<Shape, Stride> const& layout) {
  if constexpr (is_static<Shape>::value && is_static<Stride>::value) {
    return make_ordered_layout(layout.shape(), layout.stride());
  } else {
    return make_layout(layout.shape());
  }

  CUTE_GCC_UNREACHABLE;
}

// Make a layout of the same shape,
//   with mode-0 being colmajor then following the the mode order in layout
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_fragment_like(
    Layout<Shape, Stride> const& layout) {
  auto shape = replace<0>(layout.shape(), size<0>(layout));
  auto order = replace<0>(layout.stride(), Int<0>{});
  if constexpr (is_static<decltype(shape)>::value &&
                is_static<decltype(order)>::value) {
    return make_ordered_layout(shape, order);
  } else {
    return make_layout(layout.shape());
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_identity_layout(Shape const& shape) {
  return make_layout(shape, make_basis_like(shape));
}

//
// Operations to manipulate Layouts like a tuple of pairs
//

template <std::size_t... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto get(Layout<Shape, Stride> const& layout) {
  // Let the static_asserts in get<I>(shape|stride) catch problems
  return make_layout(get<Is...>(layout.shape()), get<Is...>(layout.stride()));
}

template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto take(Layout<Shape, Stride> const& layout) {
  // Let the static_asserts in take<B,E>(shape|stride) catch problems
  return make_layout(take<B, E>(layout.shape()), take<B, E>(layout.stride()));
}

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto flatten(Layout<Shape, Stride> const& layout) {
  return make_layout(flatten(layout.shape()), flatten(layout.stride()));
}

//
// Utilities
//

// Return the layout of a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr decltype(auto) layout(
    Layout<Shape, Stride> const& layout) {
  if constexpr (sizeof...(Is) == 0) {
    return layout;
  } else {
    return get<Is...>(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

// Return the shape of a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr decltype(auto) shape(Layout<Shape, Stride>& layout) {
  return layout.template shape<Is...>();
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr decltype(auto) shape(
    Layout<Shape, Stride> const& layout) {
  return layout.template shape<Is...>();
}

// Return the stride of a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr decltype(auto) stride(
    Layout<Shape, Stride>& layout) {
  return layout.template stride<Is...>();
}

template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr decltype(auto) stride(
    Layout<Shape, Stride> const& layout) {
  return layout.template stride<Is...>();
}

// Return the number of elements in a mode
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto size(Layout<Shape, Stride> const& layout) {
  return size(shape<Is...>(layout));
}

// Return the number of modes
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto rank(Layout<Shape, Stride> const& layout) {
  return rank(shape<Is...>(layout));
}

// Return the depth of the layout
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto depth(Layout<Shape, Stride> const& layout) {
  return depth(shape<Is...>(layout));
}

// Return the codomain size of a mode
// @return M smallest integer such that @a sub_layout(c) < M for all c < size(@a
// sub_layout)
//           where sub_layout = get<Is...>(layout).
template <int... Is, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto cosize(Layout<Shape, Stride> const& layout) {
  // Protect against negative strides
  auto abs_sub_layout = make_layout(
      shape<Is...>(layout), transform_leaf(stride<Is...>(layout), abs_fn{}));
  return abs_sub_layout(size(abs_sub_layout) - Int<1>{}) + Int<1>{};
}

template <class Layout>
using cosize_t = decltype(cosize(std::declval<Layout>()));

template <class Layout>
static constexpr int cosize_v = cosize_t<Layout>::value;

// Equality
// Return a static or dynamic boolean
template <class ShapeA, class StrideA, class ShapeB, class StrideB>
CUTE_HOST_DEVICE constexpr auto operator==(
    Layout<ShapeA, StrideA> const& layoutA,
    Layout<ShapeB, StrideB> const& layoutB) {
  return layoutA.shape() == layoutB.shape() &&
         layoutA.stride() == layoutB.stride();
}

// With crd2idx(coord, shape), makes sense to have crd2idx(coord, Layout) as
// well
template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto crd2idx(Coord const& c,
                                        Layout<Shape, Stride> const& layout) {
  return crd2idx(c, layout.shape(), layout.stride());
}

//
// Slice and Dice a layout
//

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto slice(Coord const& c,
                                      Layout<Shape, Stride> const& layout) {
  return make_layout(slice(c, layout.shape()), slice(c, layout.stride()));
}

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto slice_and_offset(
    Coord const& c, Layout<Shape, Stride> const& layout) {
  return cute::make_tuple(slice(c, layout), crd2idx(c, layout));
}

template <class Coord, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto dice(Coord const& c,
                                     Layout<Shape, Stride> const& layout) {
  return make_layout(dice(c, layout.shape()), dice(c, layout.stride()));
}

//
// Transform the modes of a layout
//

namespace detail {

template <class Tuple, class F, int... I>
CUTE_HOST_DEVICE constexpr auto transform_layout(Tuple const& t,
                                                 F&& f,
                                                 seq<I...>) {
  return make_layout(f(get<I>(t))...);
}

template <class Tuple0, class Tuple1, class F, int... I, int... I0, int... I1>
CUTE_HOST_DEVICE constexpr auto transform_layout(Tuple0 const& t0,
                                                 Tuple1 const& t1,
                                                 F&& f,
                                                 seq<I...>,
                                                 seq<I0...>,
                                                 seq<I1...>) {
  return make_layout(
      f(get<I>(t0), get<I>(t1))..., get<I0>(t0)..., get<I1>(t1)...);
}

}  // end namespace detail

template <class Tuple, class F>
CUTE_HOST_DEVICE constexpr auto transform_layout(Tuple const& t, F&& f) {
  return detail::transform_layout(t, f, make_seq<decltype(rank(t))::value>{});
}

template <class Tuple0, class Tuple1, class F>
CUTE_HOST_DEVICE constexpr auto transform_layout(Tuple0 const& t0,
                                                 Tuple1 const& t1,
                                                 F&& f) {
  constexpr int R0 = decltype(rank(t0))::value;
  constexpr int R1 = decltype(rank(t1))::value;
  constexpr int R = (R0 < R1) ? R0 : R1;
  return detail::transform_layout(
      t0, t1, f, make_seq<R>{}, make_range<R, R0>{}, make_range<R, R1>{});
}

//
// Coalesce and Filter
//

namespace detail {

// Look at each element and the front of the stack (in order of priority)
// front(NewLayout)  get<I>(Layout)
//      s0:d0           _1:d1     =>  continue
//      _1:d0           s1:d1     =>  replace_front    s1:d1
//      s0:s1*d1        s1:d1     =>  replace_front s0*s1:d1
//      s0:d0           s1:d1     =>  prepend          s1:d1
//
// @pre OldShape and OldStride are flat
template <int I,
          class OldShape,
          class OldStride,
          class NewShape,
          class NewStride>
CUTE_HOST_DEVICE constexpr auto bw_coalesce(OldShape const& old_shape,
                                            OldStride const& old_stride,
                                            NewShape const& new_shape,
                                            NewStride const& new_stride) {
  if constexpr (I == -1) {
    // Base case, we're done
    if constexpr (is_constant<1, NewShape>::value) {
      return Layout<_1, _0>{};
    } else {
      return Layout<NewShape, NewStride>{new_shape, new_stride};
    }
  } else if constexpr (is_constant<1, decltype(get<I>(old_shape))>::value) {
    // shape<I>(layout) == _1, skip it and continue
    return bw_coalesce<I - 1>(old_shape, old_stride, new_shape, new_stride);
  } else if constexpr (is_constant<1, NewShape>::value) {
    // Replace our shape-1 with anything (Can only happen on input
    // new_shape/new_stride)
    return bw_coalesce<I - 1>(
        old_shape, old_stride, get<I>(old_shape), get<I>(old_stride));
  } else if constexpr (is_constant<true,
                                   decltype(get<I>(old_shape) *
                                                get<I>(old_stride) ==
                                            get<0>(new_stride))>::value) {
    // Merge modes because the shapes and strides match
    return bw_coalesce<I - 1>(
        old_shape,
        old_stride,
        replace_front(new_shape, get<I>(old_shape) * get<0>(new_shape)),
        replace_front(new_stride, get<I>(old_stride)));
  } else {
    // Can't replace or merge, so prepend a new mode
    return bw_coalesce<I - 1>(old_shape,
                              old_stride,
                              prepend(new_shape, get<I>(old_shape)),
                              prepend(new_stride, get<I>(old_stride)));
  }

  CUTE_GCC_UNREACHABLE;
}

}  // end namespace detail

// Combine all the modes that are possible to combine
// Does not respect the profile of the layout, but does preserve total size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto coalesce(Layout<Shape, Stride> const& layout) {
  auto flat_shape = flatten(layout.shape());
  auto flat_stride = flatten(layout.stride());

  constexpr int R = decltype(rank(flat_shape))::value;
  return detail::bw_coalesce<R - 2>(
      flat_shape, flat_stride, get<R - 1>(flat_shape), get<R - 1>(flat_stride));
}

// Apply coalesce at the terminals of trg_profile
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto coalesce(Layout<Shape, Stride> const& layout,
                                         IntTuple const& trg_profile) {
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<Shape, Stride>::rank);
    return transform_layout(
        layout, trg_profile, [](auto const& l, auto const& t) {
          return coalesce(l, t);
        });
  } else {
    return coalesce(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

// Replace the modes in layout that have a 0-stride with a 1-size
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto filter_zeros(
    Layout<Shape, Stride> const& layout) {
  return make_layout(filter_zeros(layout.stride(), layout.shape()),
                     layout.stride());
}

// Remove all of the 0-strides and 1-sizes
// Return 1-shape if empty
template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto filter(Layout<Shape, Stride> const& layout) {
  return coalesce(filter_zeros(layout));
}

// Apply filter at the terminals of trg_profile
template <class Shape, class Stride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto filter(Layout<Shape, Stride> const& layout,
                                       IntTuple const& trg_profile) {
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<Shape, Stride>::rank);
    return transform_layout(
        layout, trg_profile, [](auto const& l, auto const& t) {
          return filter(l, t);
        });
  } else {
    return filter(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Append, Prepend, Replace
//

template <int N,
          class ShapeA,
          class StrideA,
          class ShapeX = _1,
          class StrideX = _0>
CUTE_HOST_DEVICE constexpr auto append(Layout<ShapeA, StrideA> const& layout,
                                       Layout<ShapeX, StrideX> const& x = {}) {
  return make_layout(append<N>(layout.shape(), x.shape()),
                     append<N>(layout.stride(), x.stride()));
}

template <int N,
          class ShapeA,
          class StrideA,
          class ShapeX = _1,
          class StrideX = _0>
CUTE_HOST_DEVICE constexpr auto prepend(Layout<ShapeA, StrideA> const& layout,
                                        Layout<ShapeX, StrideX> const& x = {}) {
  return make_layout(prepend<N>(layout.shape(), x.shape()),
                     prepend<N>(layout.stride(), x.stride()));
}

template <int N, class ShapeA, class StrideA, class ShapeX, class StrideX>
CUTE_HOST_DEVICE constexpr auto replace(Layout<ShapeA, StrideA> const& layout,
                                        Layout<ShapeX, StrideX> const& x) {
  return make_layout(replace<N>(layout.shape(), x.shape()),
                     replace<N>(layout.stride(), x.stride()));
}

template <int B, int E, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto group(Layout<Shape, Stride> const& layout) {
  return make_layout(group<B, E>(layout.shape()), group<B, E>(layout.stride()));
}

//
// Composition of two layouts: lhs o rhs
// @post compatible(rhs, result)
// @post result(c) = lhs(rhs(c))
//         for all c in the domain of result
//

namespace detail {

template <class LShape, class LStride, class RShape, class RStride>
CUTE_HOST_DEVICE constexpr auto composition(Layout<LShape, LStride> const& lhs,
                                            RShape const& rhs_shape,
                                            RStride const& rhs_stride) {
  if constexpr (is_tuple<RShape>::value) {
    // Apply the right-distributivity of Layout composition
    return transform_layout(
        rhs_shape, rhs_stride, [&](auto const& s, auto const& d) {
          return composition(lhs, s, d);
        });
  } else if constexpr (is_scaled_basis<RStride>::value) {
    // Special case for a ScaledBasis stride
    return composition(
        get<rhs_stride.mode()>(lhs), rhs_shape, rhs_stride.value());
  } else if constexpr (is_integral<RStride>::value) {
    // Integral Rstride (and RShape)

    // NOTE: Should only flatten once for efficiency
    auto flat_shape = flatten(lhs.shape());
    auto flat_stride = flatten(lhs.stride());
    [[maybe_unused]] constexpr int R = rank(flat_shape);

    if constexpr (is_constant<0, RStride>::value) {
      // Special case shortcut for any static stride-0
      return Layout<RShape, RStride>{rhs_shape, rhs_stride};
    } else if constexpr (is_integral<decltype(flat_shape)>::value) {
      // Special case shortcut for any integral LShape
      auto result_stride = rhs_stride * flat_stride;
      return Layout<RShape, decltype(result_stride)>{rhs_shape, result_stride};
    } else if constexpr (is_constant<1, RStride>::value) {
      // Special case shortcut for any static stride-1
      auto result_shape_0 = take<0, R - 1>(flat_shape);

      // Mod out the rhs_shape from the lhs.shape()
      auto const [result_shape_1, rest_shape] =
          fold(result_shape_0,
               make_tuple(make_tuple(), rhs_shape),
               [](auto const& init, auto const& si) {
                 return make_tuple(
                     append(get<0>(init), cute::min(abs(si), get<1>(init))),
                     shape_div(get<1>(init), abs(si)));
               });

      // Jump into coalesce and append (rest_shape, get<R-1>(lhs.stride())
      return detail::bw_coalesce<R - 2>(
          result_shape_1, flat_stride, rest_shape, get<R - 1>(flat_stride));
    } else {
      // General case
      auto result_shape_0 = take<0, R - 1>(flat_shape);
      auto result_stride_0 = take<0, R - 1>(flat_stride);

      // Divide out the rhs_stride from the lhs.shape()
      auto const [result_shape_1, rest_stride] = fold(
          result_shape_0,
          make_tuple(make_tuple(), rhs_stride),
          [](auto const& init, auto const& di) {
            return make_tuple(append(get<0>(init), shape_div(di, get<1>(init))),
                              shape_div(get<1>(init), di));
          });

      // Apply any lhs.shape() changes to the stride
      auto result_stride_1 = elem_scale(
          result_stride_0, shape_div(result_shape_0, result_shape_1));

      // Mod out the rhs_shape from the lhs.shape()
      auto const [result_shape_2, rest_shape] =
          fold(result_shape_1,
               make_tuple(make_tuple(), rhs_shape),
               [](auto const& init, auto const& si) {
                 return make_tuple(
                     append(get<0>(init), cute::min(abs(si), get<1>(init))),
                     shape_div(get<1>(init), abs(si)));
               });

      // Jump into coalesce and append (rest_shape, rest_stride *
      // get<R-1>(lhs.stride())
      return detail::bw_coalesce<R - 2>(result_shape_2,
                                        result_stride_1,
                                        rest_shape,
                                        rest_stride * get<R - 1>(flat_stride));
    }
  }

  CUTE_GCC_UNREACHABLE;
}

}  // end namespace detail

template <class LShape, class LStride, class RShape, class RStride>
CUTE_HOST_DEVICE constexpr auto composition(
    Layout<LShape, LStride> const& lhs, Layout<RShape, RStride> const& rhs) {
  // return detail::composition(flatten(lhs), rhs.shape(), rhs.stride());
  return detail::composition(lhs, rhs.shape(), rhs.stride());
}

template <class LShape, class LStride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto composition(Layout<LShape, LStride> const& lhs,
                                            IntTuple const& rhs) {
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<LShape, LStride>::rank);
    // Drop any modes of lhs that aren't hit by rhs
    return detail::transform_layout(
        lhs,
        rhs,
        [](auto const& l, auto const& r) { return composition(l, r); },
        make_seq<tuple_size<IntTuple>::value>{},
        seq<>{},
        seq<>{});
  } else if constexpr (is_underscore<IntTuple>::value) {
    return lhs;
  } else {
    return composition(lhs, make_layout(rhs));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Complement
//
// Build the complement of a layout.
// @post size(@a result) >= @a cosize_hi / size(filter(@a layout)));
// @post For all i in [1,size(@a result)),
//           @a result(i) < @a result(i-1)
//           For all j in [0, size(@a layout)),
//               @a result(i) != @a layout(j)
//

template <class Shape, class Stride, class CoSizeHi>
CUTE_HOST_DEVICE constexpr auto complement(Layout<Shape, Stride> const& layout,
                                           CoSizeHi const& cosize_hi) {
  // Remove the stride-0 modes, the size-1 modes, and flatten the layout
  auto flat_layout = filter(layout);

  if constexpr (is_constant<0, decltype(flat_layout.stride())>::value) {
    // Special case for stride-0 layout
    return make_layout(cosize_hi);
  } else {
    // General case
    constexpr int R = decltype(rank(flat_layout))::value;
    static_assert(R == 1 || is_static<decltype(flat_layout.stride())>::value,
                  "Dynamic-stride complement only for rank-1 layouts");

    // Should just be a sort and a fold...
    // Then we could even handle dynamic strides (but they would destroy all
    // static strides)
    auto result = fold(
        make_seq<R - 1>{},
        make_tuple(flat_layout.shape(),
                   flat_layout.stride(),
                   make_tuple(),
                   make_tuple(Int<1>{})),
        [](auto const& init, auto i) {
          auto curr_stride = cute::min(get<1>(init));
          auto curr_idx = find(get<1>(init), curr_stride);
          auto curr_shape = get<curr_idx>(get<0>(init));

          return make_tuple(
              remove<curr_idx>(get<0>(init)),  // Remove the curr shape
              remove<curr_idx>(get<1>(init)),  // Remove the curr stride
              append(get<2>(init),
                     curr_stride /
                         get<3, i>(
                             init)),  // new shape  = curr_stride / last_stride
              append(
                  get<3>(init),
                  curr_shape *
                      curr_stride));  // new stride = curr_shape  * curr_stride
        });

    // Append the last shape mode
    auto result_stride = get<3>(result);
    auto result_shape = append(
        get<2>(result),
        get<1, 0>(result) /
            back(result_stride));  // new shape  = curr_stride / last_stride

    // Compute the rest_stride
    auto rest_stride = get<0, 0>(result) * get<1, 0>(result);
    // return make_layout(append(result_shape,  ceil_div(cosize_hi,
    // rest_stride)), append(result_stride, rest_stride));
    //  Jump into coalesce and append (ceil_div(cosize_hi, rest_stride),
    //  rest_stride)
    return detail::bw_coalesce<R - 1>(result_shape,
                                      result_stride,
                                      ceil_div(cosize_hi, rest_stride),
                                      rest_stride);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto complement(
    Layout<Shape, Stride> const& layout) {
  return complement(layout, cosize(layout));
}

//
// Right-Inverse and Left-Inverse
//

namespace detail {

template <int I, class Shape, class Stride, int... Is>
CUTE_HOST_DEVICE constexpr auto inverse_seq(Shape const& shape,
                                            Stride const& stride,
                                            seq<Is...>) {
  if constexpr (I == decltype(rank(stride))::value) {
    return seq<Is...>{};
  } else {
    // auto next_stride = get<I>(shape) * get<I>(stride);
    using next_stride =
        decltype(get<I>(shape) * get<I>(stride));  // NOTE: WAR for g++-7

    if constexpr (is_static<next_stride>::value) {
      auto next_idx = find_if(stride, [](auto a) {
        return is_constant<next_stride::value, decltype(a)>{};
      });
      return inverse_seq<next_idx>(shape, stride, seq<Is..., I>{});
    } else {
      return seq<Is..., I>{};
    }
  }

  CUTE_GCC_UNREACHABLE;
}

}  // end namespace detail

//
// Build the right-inverse of a layout
// @pre is_static<Layout>
// @result A layout @a result such that
//    @a layout(@a result(i)) == i for all i < size(@a result)
// @result A layout @a result such that
//    composition(@a layout, @a result) is identical to
//    make_layout(shape(result))
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto right_inverse(
    Layout<Shape, Stride> const& layout) {
  auto flat_layout = coalesce(layout);
  auto astride = transform_leaf(flat_layout.stride(), abs_fn{});

  // Find Int<1>{}, the starting idx, and follow the strides to gen inverse_seq
  auto next_I =
      find_if(astride, [](auto a) { return is_constant<1, decltype(a)>{}; });
  [[maybe_unused]] auto iseq =
      detail::inverse_seq<next_I>(flat_layout.shape(), astride, seq<>{});

  if constexpr (tuple_size<decltype(iseq)>::value == 0) {
    return Layout<_1, _0>{};  // Empty case, nothing found
  } else {
    // Generate the corresponding new strides and construct
    auto rstride = compact_col_major(flat_layout.shape());
    return make_layout(
        unwrap(transform(iseq, [&](auto i) { return shape<i>(flat_layout); })),
        unwrap(transform(iseq, [&](auto i) {
          return signum(stride<i>(flat_layout)) * get<i>(rstride);
        })));
  }

  CUTE_GCC_UNREACHABLE;
}

CUTE_HOST_DEVICE constexpr auto right_inverse(Underscore const& _) { return _; }

//
// Build the left-inverse of a layout
// @pre is_static<Layout>
// @pre not has_int0<Layout>   // @a layout has no 0-strides (is injective)
// @result A layout @a result such that
//    @a result(@a layout(i)) == i for all i < size(@a layout)
// @result A layout @a result such that
//    composition(@a result, @a layout) is identical to
//    make_layout(shape(layout))
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto left_inverse(
    Layout<Shape, Stride> const& layout) {
  return right_inverse(make_layout(layout, complement(layout)));
}

CUTE_HOST_DEVICE constexpr auto left_inverse(Underscore const& _) { return _; }

//
// Max Common Vector
//

/* Return Int<N> such that N is the maximum number of continguous elements
 * that logically correspond in the layouts of @a a and @a b. This is,
 * the number of elements that could reasonably be "vectorized" in the layouts.
 *
 * @returns Int<N> with N >= 1
 * @post For all 0 <= n < N, a(b[n]) == n  (NOTE: Problems with negative
 * strides/coords in this post-condition)
 */
template <class ShapeA, class StrideA, class ShapeB, class StrideB>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    Layout<ShapeA, StrideA> const& a, Layout<ShapeB, StrideB> const& b) {
  if constexpr (is_static<Layout<ShapeA, StrideA>>::value &&
                is_static<Layout<ShapeB, StrideB>>::value) {
    auto result = coalesce(composition(a, right_inverse(b)));

    if constexpr (is_constant<1, decltype(stride<0>(result))>::value) {
      return shape<0>(result);
    } else {
      return Int<1>{};
    }
  } else {
    // Dynamic case  NOTE: could weaken if we assume dynamic strides are large
    // and multiples of the vector
    return Int<1>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Zip
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto zip(Layout<Shape, Stride> const& layout) {
  return make_layout(zip(layout.shape()), zip(layout.stride()));
}

template <class TShape, class TStride, class UShape, class UStride>
CUTE_HOST_DEVICE constexpr auto zip(Layout<TShape, TStride> const& layoutA,
                                    Layout<UShape, UStride> const& layoutB) {
  return make_layout(zip(layoutA.shape(), layoutB.shape()),
                     zip(layoutA.stride(), layoutB.stride()));
}

//
// Tile unzip
//   Logical product and logical divide (on layouts) produce rank-2 results by
//   design. Follow the profile of @a tile and zip the rank-2 modes located at
//   the terminals into their own mode.
//

template <class LShape, class LStride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto tile_unzip(
    Layout<LShape, LStride> const& layout, IntTuple const& tile) {
  return make_layout(zip2_by(layout.shape(), tile),
                     zip2_by(layout.stride(), tile));
}

//
// Logical divide
//

template <class LShape, class LStride, class TShape, class TStride>
CUTE_HOST_DEVICE constexpr auto logical_divide(
    Layout<LShape, LStride> const& layout,
    Layout<TShape, TStride> const& tile) {
  // CUTE_STATIC_ASSERT_V(size(layout) % size(tile) == Int<0>{},
  //                      "Tiling does not evenly divide the block");
  //  NOTE: With tiles that have stride-0, this doesn't have to be true

  return composition(layout, make_layout(tile, complement(tile, size(layout))));
}

template <class LShape, class LStride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto logical_divide(
    Layout<LShape, LStride> const& layout, IntTuple const& tile) {
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<LShape, LStride>::rank,
                  "logical_divide: Too many modes in tile.");
    return transform_layout(layout, tile, [](auto const& l, auto const& t) {
      return logical_divide(l, t);
    });
  } else if constexpr (is_underscore<IntTuple>::value) {
    return layout;
  } else if constexpr (is_integral<IntTuple>::value) {
    return logical_divide(layout, make_layout(tile));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Convenience operator
//   that produces layouts like ((BLK_A,BLK_B,...),(a,b,...,x,y))
//   by gathering the tile modes and residuals into a rank-2 result.
//

template <class LShape, class LStride, class Tile>
CUTE_HOST_DEVICE constexpr auto zipped_divide(
    Layout<LShape, LStride> const& layout, Tile const& tile) {
  return tile_unzip(logical_divide(layout, tile), tile);
}

// Same as zipped_divide, but unpacks the second mode:
// ((BLK_A,BLK_B,...),a,b,...,x,y)
template <class LShape, class LStride, class Tile>
CUTE_HOST_DEVICE constexpr auto tiled_divide(
    Layout<LShape, LStride> const& layout, Tile const& tile) {
  auto div = zipped_divide(layout, tile);

  auto R = rank<1>(div);
  return div(_, repeat<R>(_));
}

//
// Logical product
//

template <class LShape, class LStride, class TShape, class TStride>
CUTE_HOST_DEVICE constexpr auto logical_product(
    Layout<LShape, LStride> const& layout,
    Layout<TShape, TStride> const& tile) {
  return make_layout(
      layout,
      composition(complement(layout, size(layout) * cosize(tile)), tile));
}

template <class LShape, class LStride, class IntTuple>
CUTE_HOST_DEVICE constexpr auto logical_product(
    Layout<LShape, LStride> const& layout, IntTuple const& tile) {
  if constexpr (is_tuple<IntTuple>::value) {
    static_assert(tuple_size<IntTuple>::value <= Layout<LShape, LStride>::rank);
    return transform_layout(layout, tile, [](auto const& l, auto const& t) {
      return logical_product(l, t);
    });
  } else if constexpr (is_underscore<IntTuple>::value) {
    return layout;
  } else if constexpr (is_integral<IntTuple>::value) {
    return logical_product(layout, make_layout(tile));
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Convenience operator
//   that produces layouts like ((BLK_A,BLK_B,...),(a,b,...,x,y))
//   by gathering the block modes and products into a rank-2 result.
//

template <class LShape, class LStride, class Tile>
CUTE_HOST_DEVICE constexpr auto zipped_product(
    Layout<LShape, LStride> const& layout, Tile const& tile) {
  return tile_unzip(logical_product(layout, tile), tile);
}

// Same as zipped_product, but unpacks the second mode:
// ((BLK_A,BLK_B,...),a,b,...,x,y)
template <class LShape, class LStride, class Tile>
CUTE_HOST_DEVICE constexpr auto tiled_product(
    Layout<LShape, LStride> const& layout, Tile const& tile) {
  auto div = zipped_product(layout, tile);

  auto R = rank(tile);
  return div(_, repeat<R>(_));
}

// Attempts to reproduce layout "block" over layout "layout"
// That is, think of every element of "layout" as a "block"
//   and return the layout of the resulting structure
template <class TShape, class TStride, class UShape, class UStride>
CUTE_HOST_DEVICE constexpr auto blocked_product(
    Layout<TShape, TStride> const& block,
    Layout<UShape, UStride> const& layout) {
  constexpr int R = cute::max(rank_v<TShape>, rank_v<UShape>);
  auto padded_block = append<R>(block);
  auto padded_layout = append<R>(layout);

  auto result = logical_product(padded_block, padded_layout);

  return coalesce(zip(get<0>(result), get<1>(result)), repeat<R>(Int<1>{}));
}

template <class TShape, class TStride, class UShape, class UStride>
CUTE_HOST_DEVICE constexpr auto raked_product(
    Layout<TShape, TStride> const& block,
    Layout<UShape, UStride> const& layout) {
  constexpr int R = cute::max(rank_v<TShape>, rank_v<UShape>);
  auto padded_block = append<R>(block);
  auto padded_layout = append<R>(layout);

  auto result = logical_product(padded_block, padded_layout);

  return coalesce(zip(get<1>(result), get<0>(result)), repeat<R>(Int<1>{}));
}

template <class Shape,
          class Stride,
          class TrgShape,
          class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr auto tile_to_shape(
    Layout<Shape, Stride> const& layout,
    TrgShape const& trg_shape,
    ModeOrder const& ord_shape = {}) {
  CUTE_STATIC_ASSERT_V(rank(layout) <= rank(trg_shape),
                       "Rank of layout must be <= rank of target shape.");
  constexpr int R = rank_v<TrgShape>;

  auto padded_layout = append<R>(layout);

  auto layout_shape = product_each(padded_layout.shape());
  auto target_shape = product_each(trg_shape);

  // Assert proper division
  CUTE_STATIC_ASSERT_V(
      sum(transform(target_shape, layout_shape, modulus{})) == Int<0>{},
      "Layout shape does not divide the target shape.");

  auto product_shape = shape_div(target_shape, layout_shape);

  return coalesce(
      blocked_product(padded_layout,
                      make_ordered_layout(product_shape, ord_shape)),
      product_shape);
}

//
// Upcast
//   For stride-1 mode, divide size by N. Divide all other strides by N.
//

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Shape const& shape,
                                       Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {  // tuple stride
    return transform_layout(shape, stride, [](auto const& s, auto const& d) {
      return upcast<N>(s, d);
    });
  } else if constexpr (is_constant<0, Stride>::value) {  // static-0 stride
    return Layout<Shape, Stride>{shape, stride};
  } else if constexpr (is_static<Stride>::value) {  // static stride
    return make_layout(shape_div(shape, shape_div(Int<N>{}, abs(stride))),
                       shape_div(stride, Int<N>{}));
  } else {  // dynamic stride
    // assume dynamic strides are larger than N and divisible
    // assert(stride % N == 0);
    return make_layout(shape, safe_div(stride, Int<N>{}));
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto upcast(Layout<Shape, Stride> const& layout) {
  return upcast<N>(layout.shape(), layout.stride());
}

//
// Downcast
//   For stride-1 mode, multiply size by N. Multiply all other strides by N.
//

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto downcast(Shape const& shape,
                                         Stride const& stride) {
  if constexpr (is_tuple<Shape>::value) {
    return transform_layout(shape, stride, [](auto const& s, auto const& d) {
      return downcast<N>(s, d);
    });
  } else if constexpr (is_constant<1, Stride>::value ||
                       is_constant<-1, Stride>::value) {
    return make_layout(shape * Int<N>{}, stride);
  } else {
    return make_layout(shape, stride * Int<N>{});
  }

  CUTE_GCC_UNREACHABLE;
}

template <int N, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto downcast(Layout<Shape, Stride> const& layout) {
  CUTE_STATIC_ASSERT(has_int1<Stride>::value,
                     "Downcast requires adjacent elements");
  return downcast<N>(layout.shape(), layout.stride());
}

//
// Recast
//

template <class OldType, class NewType, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto recast(Layout<Shape, Stride> const& layout) {
  if constexpr (sizeof(NewType) == sizeof(OldType)) {
    return layout;
  } else if constexpr (sizeof(NewType) > sizeof(OldType)) {
    static_assert(sizeof(NewType) % sizeof(OldType) == 0,
                  "NewType must be a multiple of OldType");
    return upcast<sizeof(NewType) / sizeof(OldType)>(layout);
  } else if constexpr (sizeof(NewType) < sizeof(OldType)) {
    static_assert(sizeof(OldType) % sizeof(NewType) == 0,
                  "NewType must be a divisor of OldType");
    return downcast<sizeof(OldType) / sizeof(NewType)>(layout);
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Display utilities
//

template <class Shape, class Stride>
CUTE_HOST_DEVICE void print(Layout<Shape, Stride> const& layout) {
  print(layout.shape());
  print(":");
  print(layout.stride());
}

template <class Shape, class Stride>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   Layout<Shape, Stride> const& layout) {
  return os << shape(layout) << ":" << stride(layout);
}

// Generic 2D Layout to console table
template <class Layout>
CUTE_HOST_DEVICE void print_layout(Layout const& layout)  // (m,n) -> idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  int idx_width = num_digits(cosize(layout)) + 2;
  const char* delim = "+-----------------------";

  print(layout);
  print("\n");

  // Column indices
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) {
    printf("  %*d ", idx_width - 2, n);
  }
  printf("\n");

  // Print out A m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    print("    ");
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%.*s", idx_width + 1, delim);
    }
    printf("+\n");
    // Values
    printf("%2d  ", m);  // Row indices
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("| %*d ", idx_width - 2, int(layout(m, n)));
    }
    printf("|\n");
  }
  // Footer
  print("    ");
  for (int n = 0; n < size<1>(layout); ++n) {
    printf("%.*s", idx_width + 1, delim);
  }
  printf("+\n");
}

// Generic ThrVal 2D Layout to console table
template <class Layout, class ThrID>
CUTE_HOST_DEVICE void print_layout(
    Layout const& layout,
    ThrID const& thrid)  // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  print(layout);
  print("\n");
  print(thrid);
  print("\n");

  // Print out m-by-n
  for (int m = 0; m < size<0>(layout); ++m) {
    // Header
    for (int n = 0; n < size<1>(layout); ++n) printf("+------");
    printf("+\n");
    // Values
    for (int n = 0; n < size<1>(layout); ++n)
      printf("|%03d-%02d",
             int(thrid(layout(m, n) % size(thrid))),
             int(layout(m, n) / size(thrid)));
    printf("|\n");
  }
  // Footer
  for (int n = 0; n < size<1>(layout); ++n) printf("+------");
  printf("+\n");
}

// Generic 2D Layout to Latex printer -- B&W 8-value color coding
template <class Layout>
CUTE_HOST_DEVICE void print_latex(Layout const& layout)  // (m,n) -> idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  char const* latex_header =
      "\\documentclass[convert]{standalone}\n"
      "\\usepackage{tikz}\n\n"
      "\\begin{document}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum "
      "size=1cm,anchor=center,font=\\Large}]\n\n";
  char const* latex_footer =
      "\\end{tikzpicture}\n"
      "\\end{document}\n";

  char const* color_map[8] = {"black!00",
                              "black!40",
                              "black!20",
                              "black!60",
                              "black!10",
                              "black!50",
                              "black!30",
                              "black!70"};

  // Header
  printf("%% Layout: ");
  print(layout);
  printf("\n");

  printf(latex_header);

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int idx = layout(i, j);

      printf("\\node[box,fill=%s] at (%d,%d) {%d};\n",
             color_map[idx % 8],
             i,
             j,
             idx);
    }
  }

  // Labels
  for (int i = 0, j = -1; i < size<0>(layout); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(layout); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  printf(latex_footer);
}

// Generic ThrVal 2D Layout to Latex TIKZ -- 8-value color coded by thread
template <class Layout, class ThrID>
CUTE_HOST_DEVICE void print_latex(
    Layout const& layout,
    ThrID const& thr)  // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});

  char const* latex_header =
      "\\documentclass[convert]{standalone}\n"
      "\\usepackage{tikz}\n\n"
      "\\begin{document}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
  char const* latex_footer =
      "\\end{tikzpicture}\n"
      "\\end{document}\n";

  char const* color_map[8] = {"{rgb,255:red,175;green,175;blue,255}",
                              "{rgb,255:red,175;green,255;blue,175}",
                              "{rgb,255:red,255;green,255;blue,175}",
                              "{rgb,255:red,255;green,175;blue,175}",
                              "{rgb,255:red,210;green,210;blue,255}",
                              "{rgb,255:red,210;green,255;blue,210}",
                              "{rgb,255:red,255;green,255;blue,210}",
                              "{rgb,255:red,255;green,210;blue,210}"};

  // Header
  printf("%% layout: ");
  print(layout);
  printf("\n");
  printf("%% thrid:  ");
  print(thr);
  printf("\n\n");

  printf(latex_header);

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int thrid = layout(i, j) % size(thr);
      int val_idx = layout(i, j) / size(thr);
      int thr_idx = thr(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i,
             j,
             thr_idx,
             val_idx);
    }
  }

  // Labels
  for (int i = 0, j = -1; i < size<0>(layout); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(layout); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  printf(latex_footer);
}

}  // end namespace cute

//
// Extended Layouts
//

#include <cute/swizzle_layout.hpp>
