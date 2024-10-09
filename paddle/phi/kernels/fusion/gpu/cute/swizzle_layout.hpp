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

#include <cute/layout.hpp>

#include <cute/swizzle.hpp>

/* This implements a ComposedLayout of the form
 *   InvolutionFn o OffsetPlus o Layout
 * where the InvolutionFn need not be linear (hence the need for the Offset).
 *
 * This ComposedLayout provides similar coordinate-to-index mapping and layout
 * manipulations, but is not considered a "normal" layout. For example, this
 * layout provides size() functions, but does not provide stride() functions.
 *
 * Furthermore, for known InvolutionFns, this layout attempts to decay itself
 *    to a normal-layout with dynamic or static strides.
 * This is possible by determining the subdomain of the Involution function
 *    that is identity and testing if the right Layout's codomain is contained
 *    within it.
 */

namespace cute {

// A Layout of non-trivially composable functions: F o I o L
template <class InvolutionFn, class IntermediateOffset, class Layout>
struct ComposedLayout : private cute::tuple<InvolutionFn,
                                            IntermediateOffset,
                                            Layout>  // EBO for static layouts
{
  CUTE_HOST_DEVICE constexpr ComposedLayout(
      InvolutionFn const& fn = {},
      IntermediateOffset const& offset = {},
      Layout const& layout = {})
      : cute::tuple<InvolutionFn, IntermediateOffset, Layout>(
            fn, offset, layout) {}

  //
  // Accessors
  //

  static constexpr int rank = Layout::rank;

  CUTE_HOST_DEVICE constexpr decltype(auto) swizzle_fn() const {
    return get<0>(static_cast<
                  cute::tuple<InvolutionFn, IntermediateOffset, Layout> const&>(
        *this));
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) offset_fn() const {
    return get<1>(static_cast<
                  cute::tuple<InvolutionFn, IntermediateOffset, Layout> const&>(
        *this));
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) layout_fn() const {
    return get<2>(static_cast<
                  cute::tuple<InvolutionFn, IntermediateOffset, Layout> const&>(
        *this));
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) layout() const { return *this; }

  CUTE_HOST_DEVICE constexpr decltype(auto) shape() const {
    return layout_fn().shape();
  }

  // Doesn't really make sense to ask for the strides of this "layout"
  CUTE_HOST_DEVICE constexpr decltype(auto) stride() const = delete;

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
      return swizzle_fn()(to_integral(offset_fn()) +
                          layout_fn()(coord));  // (F o L)(c)
    }

    CUTE_GCC_UNREACHABLE;
  }

  // Map a 1D linear coordinate to a flat ND logical coordinate
  template <class Int, __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr auto operator[](Int const& linear_idx) const {
    return get_flat_coord(linear_idx);
  }

  // Convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr auto operator()(Coord0 const& c0,
                                             Coord1 const& c1,
                                             Coords const&... cs) const {
    return operator()(make_coord(c0, c1, cs...));
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

  // NOTE Only valid for compact layouts

  // Return the (hierarchical) ND logical coordinate corresponding to the linear
  // index
  // @post this->crd2idx(@a result) == idx
  // @post congruent(@a result, shape())
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_hier_coord(IInt const& idx) const {
    return layout_fn().get_hier_coord(
        swizzle_fn()(idx) - to_integral(offset_fn()));  // (L^-1 o F)(k)
  }

  // Return the (flat) ND logical coordinate corresponding to the linear index
  // @post this->crd2idx(@a result) == idx
  // @post rank(@a result) == rank(shape()) && depth(@a result) == 1
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_flat_coord(IInt const& idx) const {
    return layout_fn().get_flat_coord(
        swizzle_fn()(idx) - to_integral(offset_fn()));  // (L^-1 o F)(k)
  }

  // Return the generalized column-major 1D logical coordinate corresponding to
  // the linear index
  // @post this->crd2idx(@a result) == idx
  // @post is_integral<decltype(@a result)>::value
  template <class IInt, __CUTE_REQUIRES(is_integral<IInt>::value)>
  CUTE_HOST_DEVICE constexpr auto get_1d_coord(IInt const& idx) const {
    return layout_fn().get_1d_coord(swizzle_fn()(idx) -
                                    to_integral(offset_fn()));  // (L^-1 o F)(k)
  }
};

template <class Fn, class Offset, class Layout>
struct is_layout<ComposedLayout<Fn, Offset, Layout>> : true_type {};

template <class T>
struct is_composed_layout : false_type {};
template <class Fn, class Offset, class Layout>
struct is_composed_layout<ComposedLayout<Fn, Offset, Layout>> : true_type {};

//
// Constructors
//

template <int B, int M, int S>
CUTE_HOST_DEVICE constexpr auto make_layout(Swizzle<B, M, S> const& sxor) {
  return composition(sxor, Layout<Int<M + B + abs(S)>, Int<1>>{});
}

template <class S, class O, class L, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_layout(ComposedLayout<S, O, L> const& a,
                                            Layout<Shape, Stride> const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), make_layout(a.layout_fn(), b));
}

template <class Shape, class Stride, class S, class O, class L>
CUTE_HOST_DEVICE constexpr auto make_layout(Layout<Shape, Stride> const& a,
                                            ComposedLayout<S, O, L> const& b) {
  return composition(
      b.swizzle_fn(), b.offset_fn(), make_layout(a, b.layout_fn()));
}

namespace detail {

template <int B,
          int M,
          int S,
          class OldShape,
          class OldStride,
          class NewShape,
          class NewStride>
CUTE_HOST_DEVICE constexpr auto transfer_swizzle(
    Layout<OldShape, OldStride> const& old_layout,
    Layout<NewShape, NewStride> const& new_layout) {
  // Our goal is to determine a new swizzle for the strides in new_layout for
  // consistent vectorizations

  // This is accomplished by identifying
  //  S o L  :=:  S? o L*
  // We identify the "active" portion of S by computing (P o L)(c*) where P is a
  // projection generated by S Then that active identifier is transformed
  // through the layouts:
  //  L*(L[(P o L)(c*)])
  // which is a new swizzle identifier for S?, the new swizzle

  // Projections of the swizzle layout for composition, P
  auto swizzle_only_zy = make_layout(make_shape(Int<(1 << M)>{},
                                                Int<(1 << B)>{},
                                                Int<(1 << (abs(S) - B))>{},
                                                Int<(1 << B)>{},
                                                Int<1>{}),
                                     make_stride(Int<0>{},
                                                 Int<(1 << M)>{},
                                                 Int<0>{},
                                                 Int<(1 << (M + abs(S)))>{},
                                                 Int<0>{}));

  // Compose with the tile to get the swizzle projection, P o L  [The Z and Y
  // contributing portions of L]
  auto layout_only_zy = composition(swizzle_only_zy, old_layout);
  // Transform the end coordinate to get the active bits of the swizzle, (P o
  // L)(c*)
  auto swizzle_active_bits = layout_only_zy(size(layout_only_zy) - Int<1>{});

  // Get the Z bit and the Y bits -- keep only those that are active in Z *and*
  // Y
  auto zzz_msk = typename Swizzle<B, M, S>::zzz_msk{};
  auto yyy_msk = typename Swizzle<B, M, S>::yyy_msk{};
  auto msk_sft = typename Swizzle<B, M, S>::msk_sft{};
  auto active_Z =
      swizzle_active_bits & shiftr(swizzle_active_bits, msk_sft) & zzz_msk;
  auto active_Y =
      swizzle_active_bits & shiftr(swizzle_active_bits, -msk_sft) & yyy_msk;

  // Pass the identifiers through the old layout and new layout to make a new
  // swizzle identifier, L*(L[(P o L)(c*)])
  auto new_active_Z = new_layout(old_layout.get_1d_coord(active_Z));
  auto new_active_Y = new_layout(old_layout.get_1d_coord(active_Y));

  // Use this new swizzle identifier to construct the new swizzle for new_layout
  //   (this also makes sure it's a "valid" swizzle that Swizzle can represent)
  return composition(make_swizzle<new_active_Y, new_active_Z>(), new_layout);
}

}  // end namespace detail

template <int B, int M, int S, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto make_fragment_like(
    ComposedLayout<Swizzle<B, M, S>, Offset, Layout<Shape, Stride>> const&
        layout) {
  return detail::transfer_swizzle<B, M, S>(
      layout.layout_fn(), make_fragment_like(layout.layout_fn()));
}

//
// Utilities
//

// Return the layout of a mode
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) layout(
    ComposedLayout<Swizzle, Offset, Layout> const& clayout) {
  return composition(clayout.swizzle_fn(),
                     clayout.offset_fn(),
                     layout<Is...>(clayout.layout_fn()));
}

// Return the shape of a mode
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) shape(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return shape<Is...>(layout.layout_fn());
}

// Doesn't make sense to directly ask for the strides of this "layout"
template <int... Is, class Fn, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) stride(
    ComposedLayout<Fn, Offset, Layout> const& layout) = delete;

// Return the number of elements in a mode
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) size(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return size<Is...>(layout.layout_fn());
}

// Return the number of modes
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto rank(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return rank<Is...>(layout.layout_fn());
}

// Return the depth of the layout
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto depth(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return depth<Is...>(layout.layout_fn());
}

// Return the codomain size of a mode
template <int... Is, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto cosize(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return cosize<Is...>(layout.layout_fn());
}

//
// Operations to manipulate Layouts like a tuple of pairs
//

template <std::size_t I, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto get(
    ComposedLayout<Swizzle, Offset, Layout> const& a) {
  return composition(a.swizzle_fn(), a.offset_fn(), get<I>(a.layout_fn()));
}

template <int B, int E, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto take(
    ComposedLayout<Swizzle, Offset, Layout> const& a) {
  return composition(a.swizzle_fn(), a.offset_fn(), take<B, E>(a.layout_fn()));
}

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto flatten(
    ComposedLayout<Swizzle, Offset, Layout> const& a) {
  return composition(a.swizzle_fn(), a.offset_fn(), flatten(a.layout_fn()));
}

template <int N, class Swizzle, class Offset, class Layout, class X>
CUTE_HOST_DEVICE constexpr auto append(
    ComposedLayout<Swizzle, Offset, Layout> const& a, X const& x) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), append<N>(a.layout_fn(), x));
}

template <int B, int E, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto group(
    ComposedLayout<Swizzle, Offset, Layout> const& a) {
  return composition(a.swizzle_fn(), a.offset_fn(), group<B, E>(a.layout_fn()));
}

//
// Slice a ComposedLayout
//

namespace detail {

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr auto make_swizzle_strides(true_type,
                                                     IntZ const& Z,
                                                     IntY const& Y,
                                                     Offset const& offset,
                                                     int_sequence<I...>) {
  // Below is an optimized/compressed version of:
  // return make_tuple((swizzle(offset + Z*Int<(1 << I)>{}) -
  // swizzle(offset))...);
  // with knowledge of Swizzle, I... ranges for each B bits,
  //    and the layout won't slice along z-bits that are already set

  // y\z  0   1
  //   0  Z  DC
  //   1 -Z  DC

  return cute::make_tuple(
      conditional_return((offset & (Y << Int<I>{})) == Int<0>{},
                         Z << Int<I>{},
                         -(Z << Int<I>{}))...);
}

template <class IntZ, class IntY, class Offset, int... I>
CUTE_HOST_DEVICE constexpr auto make_swizzle_strides(false_type,
                                                     IntZ const& Z,
                                                     IntY const& Y,
                                                     Offset const& offset,
                                                     int_sequence<I...>) {
  // Below is an optimized/compressed version of:
  // return make_tuple((swizzle(offset + Y*Int<(1 << I)>{}) -
  // swizzle(offset))...);
  // with knowledge of Swizzle, I... ranges for each B bits,
  //    and the layout won't slice along y-bits that are already set

  // y\z  0   1
  //   0 Y+Z Y-Z
  //   1 DC  DC

  return cute::make_tuple(
      conditional_return((offset & (Z << Int<I>{})) == Int<0>{},
                         (Y + Z) << Int<I>{},
                         (Y - Z) << Int<I>{})...);
}

}  // end namespace detail

template <class Coord, int B, int M, int S, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto slice_and_offset(
    Coord const& coord,
    ComposedLayout<Swizzle<B, M, S>, Offset, Layout> const& layout) {
  if constexpr (all_underscore<Coord>::value) {
    // Skip the expensive/complicated attempt to decay to a normal layout and
    // just reshape
    return cute::make_tuple(composition(layout.swizzle_fn(),
                                        layout.offset_fn(),
                                        slice(coord, layout.layout_fn())),
                            Int<0>{});
  } else {
    // Projections of the swizzle layout for composition
    auto sw = make_layout(make_shape(Int<(1 << M)>{},
                                     Int<(1 << B)>{},
                                     Int<(1 << (abs(S) - B))>{},
                                     Int<(1 << B)>{},
                                     Int<1>{}));

    auto swizzle_anti_zy = make_layout(
        shape(sw),
        make_stride(
            stride<0>(sw), Int<0>{}, stride<2>(sw), Int<0>{}, size(sw)));
    auto swizzle_only_zy = make_layout(
        shape(sw),
        make_stride(
            Int<0>{}, stride<1>(sw), Int<0>{}, stride<3>(sw), Int<0>{}));

    // The portion of the layout that is not yet consumed
    auto sliced_layout = slice(coord, layout.layout_fn());

    // If the sliced_layout hits two bits that are swizzled together, then don't
    // attempt to decay

    // Compose with the layout to get the swizzle projection, P o L  [The Z and
    // Y contributing portions of L]
    //   (this also tests that shape/stride of layout compose with swizzle)
    auto sliced_layout_only_zy = composition(swizzle_only_zy, sliced_layout);
    // Transform the end coordinate to get the active bits of the swizzle, (P o
    // L)(c*)
    auto swizzle_active_bits =
        sliced_layout_only_zy(size(sliced_layout_only_zy) - Int<1>{});
    // Determine if any active bits collide under the swizzle
    auto hit_ZandY =
        !(swizzle_active_bits & ~layout.swizzle_fn()(swizzle_active_bits));

    // The portion of the layout that we are consuming now
    auto diced_layout = dice(coord, layout.layout_fn());
    auto diced_coord = dice(coord, coord);

    auto diced_layout_anti_zy = composition(swizzle_anti_zy, diced_layout);
    auto diced_layout_only_zy = composition(swizzle_only_zy, diced_layout);

    // New swizzle and offset
    auto swizzle = layout.swizzle_fn();
    // offset_only_zy interacts with swizzle and gets accumulated with
    // layout.offset_fn()
    //   being careful about the static/dynamic contributions from diced_layout
    //   and diced_coord
    auto offset_only_zy =
        layout.offset_fn() ^ to_mixed_bits(diced_layout_only_zy, diced_coord);
    // offset_anti_zy always gets passed through, no interaction with swizzle
    auto offset_anti_zy = diced_layout_anti_zy(diced_coord);

    // If Layout's codomain hits on         Y AND Z, then it's not reducible
    // If Layout's codomain hits on         Y XOR Z, then it's dynamic-normal
    // If Layout's codomain hits on neither Y NOR Z, then it's static-normal

    // Test the sliced layout for hit_X & hit_Y for potential decay
    if constexpr (is_constant<false, decltype(hit_ZandY)>::
                      value) {  // Hits on Y AND Z, so it's not reducible
      return cute::make_tuple(
          composition(swizzle, offset_only_zy, sliced_layout), offset_anti_zy);
    } else {  // Misses on Y or Z, so it's static-normal or dynamic-normal

      // Lowest bit of the Z and Y masks
      auto Z = typename Swizzle<B, M, S>::zzz_msk{} &
               -typename Swizzle<B, M, S>::zzz_msk{};
      auto Y = typename Swizzle<B, M, S>::yyy_msk{} &
               -typename Swizzle<B, M, S>::yyy_msk{};
      auto stride_lo = detail::make_swizzle_strides(
          Z < Y, Z, Y, offset_only_zy, make_int_sequence<B>{});
      auto stride_hi = detail::make_swizzle_strides(
          Z > Y, Z, Y, offset_only_zy, make_int_sequence<B>{});

      // Construct a (dynamic) layout that we can perform the composition with
      auto swizzle_layout =
          make_layout(make_shape(Int<(1 << M)>{},
                                 repeat<B>(Int<2>{}),
                                 Int<(1 << (abs(S) - B))>{},
                                 repeat<B>(Int<2>{}),
                                 Int<1>{}),
                      make_stride(Int<1>{},
                                  stride_lo,
                                  Int<(1 << (M + B))>{},
                                  stride_hi,
                                  Int<(1 << (M + B + abs(S)))>{}));

      // Decay to a normal layout with offset
      return cute::make_tuple(
          composition(swizzle_layout, sliced_layout),
          swizzle(to_integral(offset_only_zy)) + offset_anti_zy);
    }
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Coord, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto slice(
    Coord const& coord, ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return get<0>(slice_and_offset(coord, layout));
}

//
// composition
//

template <class Swizzle, class Offset, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto composition(
    Swizzle const& sxor,
    Offset const& offset,
    Layout<Shape, Stride> const& layout) {
  return ComposedLayout<Swizzle, Offset, Layout<Shape, Stride>>{
      sxor, offset, layout};
}

template <class Swizzle,
          class Offset,
          class Swizzle2,
          class Offset2,
          class Layout>
CUTE_HOST_DEVICE constexpr auto composition(
    Swizzle const& sxor,
    Offset const& offset,
    ComposedLayout<Swizzle2, Offset2, Layout> const& layout) {
  // Assume disjoint swizzles and offsets for commutivity
  return composition(composition(sxor, layout.swizzle_fn()),
                     offset ^ layout.offset_fn(),
                     layout.layout_fn());
}

// Ignore identity case
template <int M, int S, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto composition(
    Swizzle<0, M, S> const&,
    Int<0> const&,
    Layout<Shape, Stride> const& layout) {
  return layout;
}

template <int B, int M, int S, class Shape, class Stride>
CUTE_HOST_DEVICE constexpr auto composition(
    Swizzle<B, M, S> const& sxor, Layout<Shape, Stride> const& layout) {
  return composition(sxor, Int<0>{}, layout);
}

template <class SwizzleA, class OffsetA, class LayoutA, class LayoutOrTile>
CUTE_HOST_DEVICE constexpr auto composition(
    ComposedLayout<SwizzleA, OffsetA, LayoutA> const& a,
    LayoutOrTile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), composition(a.layout_fn(), b));
}

template <class ShapeA, class StrideA, int B, int M, int S>
CUTE_HOST_DEVICE constexpr auto composition(Layout<ShapeA, StrideA> const& a,
                                            Swizzle<B, M, S> const& b) {
  // Get the Z bits and the Y bits
  auto active_Y = a(typename Swizzle<B, M, S>::yyy_msk{});
  auto active_Z = a(typename Swizzle<B, M, S>::zzz_msk{});

  // Works in simple cases... but could be greatly generalized

  return composition(make_swizzle<active_Y, active_Z>(), a);
}

template <class ShapeA,
          class StrideA,
          class SwizzleB,
          class OffsetB,
          class LayoutB>
CUTE_HOST_DEVICE constexpr auto composition(
    Layout<ShapeA, StrideA> const& a,
    ComposedLayout<SwizzleB, OffsetB, LayoutB> const& b) {
  CUTE_STATIC_ASSERT_V(b.offset_fn() == Int<0>{},
                       "Require Swizzle offset == 0.");

  return composition(composition(a, b.swizzle_fn()), b.layout_fn());
}

template <class SwizzleA,
          class OffsetA,
          class LayoutA,
          class SwizzleB,
          class OffsetB,
          class LayoutB>
CUTE_HOST_DEVICE constexpr auto composition(
    ComposedLayout<SwizzleA, OffsetA, LayoutA> const& a,
    ComposedLayout<SwizzleB, OffsetB, LayoutB> const& b) {
  auto asb = composition(a.layout_fn(), b);

  return composition(composition(a.swizzle_fn(), asb.swizzle_fn()),
                     asb.offset_fn(),
                     asb.layout_fn());
}

//
// complement
//

template <class Swizzle, class Offset, class Layout, class CoSizeHi>
CUTE_HOST_DEVICE constexpr auto complement(
    ComposedLayout<Swizzle, Offset, Layout> const& layout,
    CoSizeHi const& cosize_hi) {
  // Assume there is no swizzle component in the complement
  return complement(layout.layout_fn(), cosize_hi);
}

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto complement(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return complement(layout, cosize(layout));
}

//
// inverse
//

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto right_inverse(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  CUTE_STATIC_ASSERT_V(layout.offset_fn() == Int<0>{}, "Requires 0-offset.");
  return composition(right_inverse(layout.layout_fn()), layout.swizzle_fn());
}

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto left_inverse(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  CUTE_STATIC_ASSERT_V(layout.offset_fn() == Int<0>{}, "Requires 0-offset.");
  return composition(left_inverse(layout.layout_fn()), layout.swizzle_fn());
}

//
// Other operations
//

template <int B,
          int M,
          int S,
          class Offset,
          class SLayout,
          class Shape,
          class Stride>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    ComposedLayout<Swizzle<B, M, S>, Offset, SLayout> const& a,
    Layout<Shape, Stride> const& b) {
  // This assumes that Offset is in the YZ domain of the Swizzle...
  return cute::min(Int<(1 << M)>{}, max_common_vector(a.layout_fn(), b));
}

template <class Shape,
          class Stride,
          int B,
          int M,
          int S,
          class Offset,
          class SLayout>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    Layout<Shape, Stride> const& a,
    ComposedLayout<Swizzle<B, M, S>, Offset, SLayout> const& b) {
  return max_common_vector(b, a);
}

template <int B0,
          int M0,
          int S0,
          class Offset0,
          class SLayout0,
          int B1,
          int M1,
          int S1,
          class Offset1,
          class SLayout1>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    ComposedLayout<Swizzle<B0, M0, S0>, Offset0, SLayout0> const& a,
    ComposedLayout<Swizzle<B1, M1, S1>, Offset1, SLayout1> const& b) {
  auto result = coalesce(composition(a, right_inverse(b)));

  if constexpr (is_constant<1,
                            decltype(stride<0>(result.layout_fn()))>::value) {
    return shape<0>(result);
  } else {
    return Int<1>{};
  }

  CUTE_GCC_UNREACHABLE;
}

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto zip(
    ComposedLayout<Swizzle, Offset, Layout> const& a) {
  return composition(a.swizzle_fn(), a.offset_fn(), zip(a.layout_fn()));
}

// Partitions

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto logical_divide(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), logical_divide(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto tile_unzip(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), tile_unzip(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto tiled_divide(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), tiled_divide(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto zipped_divide(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), zipped_divide(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto logical_product(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), logical_product(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto tiled_product(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), tiled_product(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto blocked_product(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), blocked_product(a.layout_fn(), b));
}

template <class Swizzle, class Offset, class LayoutA, class Tile>
CUTE_HOST_DEVICE constexpr auto raked_product(
    ComposedLayout<Swizzle, Offset, LayoutA> const& a, Tile const& b) {
  return composition(
      a.swizzle_fn(), a.offset_fn(), raked_product(a.layout_fn(), b));
}

template <class Swizzle,
          class Offset,
          class LayoutA,
          class Shape,
          class ModeOrder = GenColMajor>
CUTE_HOST_DEVICE constexpr auto tile_to_shape(
    ComposedLayout<Swizzle, Offset, LayoutA> const& layout,
    Shape const& trg_shape,
    ModeOrder const& ord_shape = {}) {
  return composition(layout.swizzle_fn(),
                     layout.offset_fn(),
                     tile_to_shape(layout.layout_fn(), trg_shape, ord_shape));
}

template <class Swizzle, class Offset, class LayoutA, class Shape>
CUTE_HOST_DEVICE constexpr auto filter(
    ComposedLayout<Swizzle, Offset, LayoutA> const& layout,
    Shape const& trg_profile) {
  return composition(layout.swizzle_fn(),
                     layout.offset_fn(),
                     filter(layout.layout_fn(), trg_profile));
}

template <class Swizzle, class Offset, class LayoutA>
CUTE_HOST_DEVICE constexpr auto coalesce(
    ComposedLayout<Swizzle, Offset, LayoutA> const& layout) {
  return composition(
      layout.swizzle_fn(), layout.offset_fn(), coalesce(layout.layout_fn()));
}

template <class Swizzle, class Offset, class LayoutA, class Shape>
CUTE_HOST_DEVICE constexpr auto coalesce(
    ComposedLayout<Swizzle, Offset, LayoutA> const& layout,
    Shape const& trg_profile) {
  return composition(layout.swizzle_fn(),
                     layout.offset_fn(),
                     coalesce(layout.layout_fn(), trg_profile));
}

///////////////////////////////////////////////////////////////////////////////
// ComposedLayout as second argument is often more difficult...

template <class Shape,
          class Stride,
          int B,
          int M,
          int S,
          class Offset,
          class LayoutT>
CUTE_HOST_DEVICE constexpr auto logical_product(
    Layout<Shape, Stride> const& block,
    ComposedLayout<Swizzle<B, M, S>, Offset, LayoutT> const& tile) {
  CUTE_STATIC_ASSERT_V(tile.offset_fn() == Int<0>{},
                       "Require Swizzle offset == 0.");
  // The new layout -- if swizzle wasn't an issue, this is the result
  //   our goal is to determine a new swizzle for these strides
  auto new_layout = logical_product(block, tile.layout_fn());

  // This is accomplished by identifying
  //  S o L  :=:  S? o L*
  // We identify the "active" portion of S by computing (P o L)(c*) where P is a
  // projection generated by S Then that active identifier is transformed
  // through the layouts:
  //  L*(L[(P o L)(c*)])
  // which is a new swizzle identifier for S?, the new swizzle

  // Projections of the swizzle layout for composition, P
  auto swizzle_only_zy = make_layout(make_shape(Int<(1 << M)>{},
                                                Int<(1 << B)>{},
                                                Int<(1 << (abs(S) - B))>{},
                                                Int<(1 << B)>{},
                                                Int<1>{}),
                                     make_stride(Int<0>{},
                                                 Int<(1 << M)>{},
                                                 Int<0>{},
                                                 Int<(1 << (M + abs(S)))>{},
                                                 Int<0>{}));

  // Compose with the tile to get the swizzle projection, P o L  [The Z and Y
  // contributing portions of L]
  auto layout_only_zy = composition(swizzle_only_zy, tile.layout_fn());
  // Transform the end coordinate to get the active bits of the swizzle, (P o
  // L)(c*)
  auto swizzle_active_bits = layout_only_zy(size(layout_only_zy) - Int<1>{});
  // Get the Z bit and the Y bits
  auto active_Z = swizzle_active_bits & typename Swizzle<B, M, S>::zzz_msk{};
  auto active_Y = swizzle_active_bits & typename Swizzle<B, M, S>::yyy_msk{};

  // Pass the identifiers through the old layout and new layout to make a new
  // swizzle identifier, L*(L[(P o L)(c*)])
  auto new_active_Z = new_layout(Int<0>{}, tile.layout_fn()[active_Z]);
  auto new_active_Y = new_layout(Int<0>{}, tile.layout_fn()[active_Y]);

  // Use this new swizzle identifier to construxt the new swizzle for new_layout
  //   (this also makes sure it's a "valid" swizzle that Swizzle can represent)
  return composition(make_swizzle<new_active_Y, new_active_Z>(), new_layout);
}

template <class Shape, class Stride, class Swizzle, class Offset, class LayoutT>
CUTE_HOST_DEVICE constexpr auto tiled_product(
    Layout<Shape, Stride> const& block,
    ComposedLayout<Swizzle, Offset, LayoutT> const& tile) {
  /// Avoid swizzle slice
  auto result = logical_product(block, tile);
  return composition(result.swizzle_fn(),
                     result.offset_fn(),
                     result.layout_fn()(_, repeat<rank_v<LayoutT>>(_)));
}

template <class Shape, class Stride, class Swizzle, class Offset, class LayoutT>
CUTE_HOST_DEVICE constexpr auto blocked_product(
    Layout<Shape, Stride> const& block,
    ComposedLayout<Swizzle, Offset, LayoutT> const& layout) {
  constexpr int R = cute::max(rank_v<Shape>, rank_v<LayoutT>);
  auto padded_block = append<R>(block, Layout<_1, _0>{});
  auto padded_layout = append<R>(layout, Layout<_1, _0>{});

  auto result = logical_product(padded_block, padded_layout);

  return composition(
      result.swizzle_fn(),
      result.offset_fn(),
      coalesce(zip(get<0>(result.layout_fn()), get<1>(result.layout_fn())),
               repeat<R>(Int<1>{})));
}

//
// Upcast and Downcast
//

template <int N, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto upcast(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return composition(upcast<N>(layout.swizzle_fn()),
                     upcast<N>(layout.offset_fn()),
                     upcast<N>(layout.layout_fn()));
}

template <int N, class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE constexpr auto downcast(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return composition(downcast<N>(layout.swizzle_fn()),
                     downcast<N>(layout.offset_fn()),
                     downcast<N>(layout.layout_fn()));
}

template <class OldType,
          class NewType,
          class Swizzle,
          class Offset,
          class Layout>
CUTE_HOST_DEVICE constexpr auto recast(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
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

template <class Swizzle, class Offset, class Layout>
CUTE_HOST_DEVICE void print(
    ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  print(layout.swizzle_fn());
  print(" o ");
  print(layout.offset_fn());
  print(" o ");
  print(layout.layout_fn());
}

template <class Swizzle, class Offset, class Layout>
CUTE_HOST std::ostream& operator<<(
    std::ostream& os, ComposedLayout<Swizzle, Offset, Layout> const& layout) {
  return os << layout.swizzle_fn() << " o " << layout.offset_fn() << " o "
            << layout.layout_fn();
}

}  // end namespace cute
