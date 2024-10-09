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

#include <cute/container/array_aligned.hpp>
#include <cute/container/array_subbyte.hpp>
#include <cute/container/tuple.hpp>
#include <cute/container/type_list.hpp>
#include <cute/numeric/integer_sequence.hpp>
#include <cute/numeric/integral_constant.hpp>
#include <cute/util/type_traits.hpp>

#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tile.hpp>

namespace cute {

//
// Engine -- owning or non-owning data store
//

// concept Engine {
//   using value_type = ;
//   iterator begin();
// };

template <class T, int N>
using ArrayEngine = typename std::conditional<(sizeof_bits<T>::value % 8 == 0),
                                              array_aligned<T, N>,
                                              array_subbyte<T, N>>::type;

template <class Iterator>
struct ViewEngine {
  using value_type =
      typename cute::remove_cvref<decltype(*std::declval<Iterator>())>::type;

  using iterator = Iterator;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }

  CUTE_HOST_DEVICE constexpr iterator& begin() { return storage_; }
};

template <class Iter>
struct is_rmem<ViewEngine<Iter>> : is_rmem<Iter> {};
template <class Iter>
struct is_smem<ViewEngine<Iter>> : is_smem<Iter> {};
template <class Iter>
struct is_gmem<ViewEngine<Iter>> : is_gmem<Iter> {};
template <class Iterator>
struct ConstViewEngine {
  using value_type =
      typename cute::remove_cvref<decltype(*std::declval<Iterator>())>::type;

  using iterator = Iterator;
  iterator storage_;

  CUTE_HOST_DEVICE constexpr iterator const& begin() const { return storage_; }
};

template <class Iter>
struct is_rmem<ConstViewEngine<Iter>> : is_rmem<Iter> {};
template <class Iter>
struct is_smem<ConstViewEngine<Iter>> : is_smem<Iter> {};
template <class Iter>
struct is_gmem<ConstViewEngine<Iter>> : is_gmem<Iter> {};
//
// Tensor
//

template <class Engine, class Layout>
struct Tensor {
  using value_type = typename Engine::value_type;
  // using pointer         = typename engine_traits<Engine>::pointer;
  // using const_pointer   = typename engine_traits<Engine>::const_pointer;
  // using reference       = typename engine_traits<Engine>::reference;
  // using const_reference = typename engine_traits<Engine>::const_reference;

  using engine_type = Engine;
  using layout_type = Layout;

  CUTE_HOST_DEVICE constexpr Tensor() {}

  template <class Ptr>
  CUTE_HOST_DEVICE constexpr Tensor(Ptr const& ptr, Layout const& layout)
      : rep_(layout, ptr) {}

  //
  // Accessors
  //

  static constexpr int rank = Layout::rank;

  CUTE_HOST_DEVICE constexpr decltype(auto) tensor() const { return *this; }

  CUTE_HOST_DEVICE constexpr decltype(auto) layout() const {
    return get<0>(rep_);
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) engine() const {
    return get<1>(rep_);
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) engine() { return get<1>(rep_); }

  CUTE_HOST_DEVICE constexpr decltype(auto) data() const {
    return engine().begin();
  }

  CUTE_HOST_DEVICE constexpr decltype(auto) data() { return engine().begin(); }

  CUTE_HOST_DEVICE constexpr decltype(auto) shape() const {
    return layout().shape();
  }

  CUTE_HOST_DEVICE constexpr auto size() const { return cute::size(shape()); }

  CUTE_HOST_DEVICE constexpr decltype(auto) stride() const {
    return layout().stride();
  }

  //
  // Indexing op() and op[]
  //

  // Index into this tensor like an array by computing the offset via layout()
  template <class Coord>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator[](Coord const& coord) {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator[](
      Coord const& coord) const {
    return data()[layout()(coord)];
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(Coord const& coord) {
    if constexpr (has_underscore<Coord>::value) {
      auto const& [sliced_layout, offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  template <class Coord>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(
      Coord const& coord) const {
    if constexpr (has_underscore<Coord>::value) {
      auto const& [sliced_layout, offset] = slice_and_offset(coord, layout());
      return make_tensor(data() + offset, sliced_layout);
    } else {
      return data()[layout()(coord)];
    }

    CUTE_GCC_UNREACHABLE;
  }

  // op() convenience function for multi-dimensional coordinates
  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(Coord0 const& c0,
                                                       Coord1 const& c1,
                                                       Coords const&... cs) {
    return operator()(make_coord(c0, c1, cs...));
  }

  template <class Coord0, class Coord1, class... Coords>
  CUTE_HOST_DEVICE constexpr decltype(auto) operator()(
      Coord0 const& c0, Coord1 const& c1, Coords const&... cs) const {
    return operator()(make_coord(c0, c1, cs...));
  }

  //
  // Compose
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto compose(Layouts const&... layouts) {
    return make_tensor(data(), layout().compose(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto compose(Layouts const&... layouts) const {
    return make_tensor(data(), layout().compose(layouts...));
  }

  //
  // Tile
  //

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto tile(Layouts const&... layouts) {
    return make_tensor(data(), layout().tile(layouts...));
  }

  template <class... Layouts>
  CUTE_HOST_DEVICE constexpr auto tile(Layouts const&... layouts) const {
    return make_tensor(data(), layout().tile(layouts...));
  }

  //
  // Utility
  //

  template <class Int, __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr auto get_1d_coord(Int const& linear_idx) const {
    return layout().get_1d_coord(linear_idx);
  }

  template <class Int, __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr auto get_hier_coord(Int const& linear_idx) const {
    return layout().get_hier_coord(linear_idx);
  }

  template <class Int, __CUTE_REQUIRES(is_integral<Int>::value)>
  CUTE_HOST_DEVICE constexpr auto get_flat_coord(Int const& linear_idx) const {
    return layout().get_flat_coord(linear_idx);
  }

  cute::tuple<layout_type, engine_type> rep_;
};

template <class Layout>
struct is_tensor : false_type {};
template <class Engine, class Layout>
struct is_tensor<Tensor<Engine, Layout>> : true_type {};

template <class Engine, class Layout>
struct is_rmem<Tensor<Engine, Layout>> : is_rmem<Engine> {};
template <class Engine, class Layout>
struct is_smem<Tensor<Engine, Layout>> : is_smem<Engine> {};
template <class Engine, class Layout>
struct is_gmem<Tensor<Engine, Layout>> : is_gmem<Engine> {};
//
// Make an owning Tensor that will allocate a static array
//

template <class T, class Layout, __CUTE_REQUIRES(is_layout<Layout>::value)>
CUTE_HOST_DEVICE constexpr auto make_tensor(Layout const& layout) {
  static_assert(is_static<Layout>::value,
                "Dynamic owning tensors not supported");
  using Engine = ArrayEngine<T, cosize_v<Layout>>;
  return Tensor<Engine, Layout>();
}

// e.g. make_tensor<double>(12)
template <class T,
          class LayoutArg,
          class... LayoutArgs,
          __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
CUTE_HOST_DEVICE constexpr auto make_tensor(LayoutArg const& arg,
                                            LayoutArgs const&... args) {
  return make_tensor<T>(make_layout(arg, args...));
}

//
// Make a non-owning Tensor that will use a pointer (view)
//

template <class Iterator,
          class Layout,
          __CUTE_REQUIRES(
              has_dereference<Iterator>::value&& is_layout<Layout>::value)>
CUTE_HOST_DEVICE constexpr auto make_tensor(Iterator const& iter,
                                            Layout const& layout) {
  using Engine = ViewEngine<Iterator>;
  return Tensor<Engine, Layout>(iter, layout);
}

// e.g. make_tensor(vec.data(), 12)
template <class Iterator,
          class LayoutArg,
          class... LayoutArgs,
          __CUTE_REQUIRES(not is_layout<LayoutArg>::value)>
CUTE_HOST_DEVICE constexpr auto make_tensor(Iterator const& iter,
                                            LayoutArg const& arg,
                                            LayoutArgs const&... args) {
  return make_tensor(iter, make_layout(arg, args...));
}

//
// make_tensor_like -- make a register tensor the same type and shape as another
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto make_tensor_like(
    Tensor<Engine, Layout> const& tensor) {
  using value_type = typename Tensor<Engine, Layout>::value_type;
  return make_tensor<value_type>(tensor.shape());
}

//
// make_fragment_like -- make a register tensor the same type, shape, and (if
// possible) order as another tensor
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto make_fragment_like(
    Tensor<Engine, Layout> const& tensor) {
  using value_type = typename Tensor<Engine, Layout>::value_type;
  return make_tensor<value_type>(make_layout_like(tensor.layout()));
}

//
// make_identity_tensor
//

template <class Shape>
CUTE_HOST_DEVICE constexpr auto make_identity_tensor(Shape const& shape) {
  return make_tensor(ArithmeticTupleIterator(
                         as_arithmetic_tuple(repeat_like(shape, Int<0>{}))),
                     make_identity_layout(shape));
}

//
// Utilities
//

// Return the subtensor of a mode
template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr decltype(auto) tensor(Tensor&& tensor) {
  return std::forward<Tensor>(tensor);
}

template <int I,
          int... Is,
          class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr decltype(auto) tensor(Tensor&& tensor) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     get<I, Is...>(tensor.layout()));
}

// Return the subtensor of a range of modes
template <int B,
          int E,
          class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr decltype(auto) take(Tensor&& tensor) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     take<B, E>(tensor.layout()));
}

// Return the layout of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) layout(
    Tensor<Engine, Layout> const& tensor) {
  return layout<Is...>(tensor.layout());
}

// Return the shape of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) shape(
    Tensor<Engine, Layout> const& tensor) {
  return shape<Is...>(tensor.layout());
}

// Return the stride of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) stride(
    Tensor<Engine, Layout> const& tensor) {
  return stride<Is...>(tensor.layout());
}

// Return the number of elements in a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr decltype(auto) size(
    Tensor<Engine, Layout> const& tensor) {
  return size<Is...>(tensor.layout());
}

// Return the rank of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto rank(Tensor<Engine, Layout> const& tensor) {
  return rank<Is...>(tensor.layout());
}

// Return the depth of a mode
template <int... Is, class Engine, class Layout>
CUTE_HOST_DEVICE constexpr auto depth(Tensor<Engine, Layout> const& tensor) {
  return depth<Is...>(tensor.layout());
}

//
// Operations to manipulate Tensors like a Layout
//

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto flatten(Tensor&& tensor) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     flatten(tensor.layout()));
}

template <class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto coalesce(Tensor&& tensor) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     coalesce(tensor.layout()));
}

template <class Tensor,
          class Profile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto coalesce(Tensor&& tensor,
                                         Profile const& profile) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     coalesce(tensor.layout(), profile));
}

// Group the modes [B,E) into a single mode
// e.g. group<2,4>(make_tensor<int>(Layout<Shape<_1,_2,_3,_4,_5,_6>>{}))
//      => make_tensor<int>(Layout<Shape<_1,_2,Shape<_3,_4>,_5,_6>>{})
template <int B,
          int E,
          class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto group_modes(Tensor&& tensor) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     group<B, E>(tensor.layout()));
}

//
// Recast
//

// NOTE: This is very dangerous to do
//   -- doesn't check dynamic integer divisibility
//   -- doesn't check alignment

// A tagged version for dispatching
template <class NewType,
          class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto recast(Tensor&& tensor, type_list<NewType>) {
  using OldType = typename remove_cvref_t<Tensor>::value_type;
  auto old_layout = tensor.layout();
  auto new_layout = recast<OldType, NewType>(old_layout);

  // If this is an upcast of a normal Layout with static negative strides, then
  // offset as well
  if constexpr (sizeof(OldType) < sizeof(NewType) &&
                not is_composed_layout<decltype(old_layout)>::value) {
    auto shape_diff = transform(
        flatten(old_layout.shape()), flatten(new_layout.shape()), minus{});
    auto extent_diff =
        transform(shape_diff, flatten(old_layout.stride()), multiplies{});
    auto offset = fold(extent_diff, Int<0>{}, [](auto const& i, auto const& a) {
      return i + cute::min(a, Int<0>{});
    });

    return make_tensor(
        recast<NewType>(std::forward<Tensor>(tensor).data() + offset),
        new_layout);
  } else {
    return make_tensor(recast<NewType>(std::forward<Tensor>(tensor).data()),
                       new_layout);
  }

  CUTE_GCC_UNREACHABLE;
}

template <class NewType,
          class Tensor,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto recast(Tensor&& tensor) {
  return recast(std::forward<Tensor>(tensor), type_list<NewType>{});
}

//
// max_common_vector
//

/* Return Int<N> such that N is the maximum number of continguous elements
 * that logically correspond in the tensors of @a a and @a b. This is,
 * the number of elements that could reasonably be vectorized into a single
 * load/store.
 *
 * @returns Int<N> with N >= 0
 *
 * A return value of Int<0> indicates that no such conclusion can be made and no
 * vectorization should be attempted.
 */
template <class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTE_HOST_DEVICE constexpr auto max_common_vector(
    Tensor<SrcEngine, SrcLayout> const& a,
    Tensor<DstEngine, DstLayout> const& b) {
  using SrcType = typename Tensor<SrcEngine, SrcLayout>::value_type;
  using DstType = typename Tensor<DstEngine, DstLayout>::value_type;

  using SrcRef = decltype(*(a.data()));
  using DstRef = decltype(*(b.data()));

  // Determine if vectorization candidates at all
  if constexpr (  // Should be the same value_types, else the copy is also
                  // performing a cast
      sizeof(SrcType) == sizeof(DstType) &&
      // The types should be trivially copyable so that vectorization is valid
      std::is_trivially_copyable<SrcType>::value &&
      std::is_trivially_copyable<DstType>::value &&
      // Should be load/storing real data, rather than implicit iterators or
      // such
      std::is_reference<SrcRef>::value && std::is_reference<DstRef>::value) {
    return max_common_vector(a.layout(), b.layout());
  } else {
    return Int<0>{};
  }

  CUTE_GCC_UNREACHABLE;
}

//
// Key algebraic operations
//

template <class Tensor,
          class Tile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto logical_divide(Tensor&& tensor,
                                               Tile const& tile) {
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     logical_divide(tensor.layout(), tile));
}

// zipped_divide is logical_divide with modes gathered into standard form
// ((BLK_A,BLK_B),(a,b))
template <class Tensor,
          class Tile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto zipped_divide(
    Tensor&& tensor,
    Tile const& tile)  // Layout or Tile<Layout...>
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     zipped_divide(tensor.layout(), tile));
}

// tiled_divide is logical_divide with the second output mode flattened
// ((BLK_A,BLK_B),a,b)
template <class Tensor,
          class Tile,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto tiled_divide(
    Tensor&& tensor,
    Tile const& tile)  // Layout or Tile<Layout...>
{
  return make_tensor(std::forward<Tensor>(tensor).data(),
                     tiled_divide(tensor.layout(), tile));
}

// logical_product on a Tensor doesn't make sense since it often increases
// cosize

//
// Logicial Divide utilities: local_partition and local_tile
//

template <class Tensor,
          class Tile,
          class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto local_partition(Tensor&& tensor,
                                                Tile const& tile,
                                                Coord const& coord) {
  constexpr int R1 = decltype(rank(tensor))::value;

  // Split the modes of tensor according to the modes of tile
  // zipped_divide returns something like ((VEC_A,VEC_B,...),(a,b,...))

  // The_coord is the coord into the first mode, flatten the rest
  return zipped_divide(std::forward<Tensor>(tensor), tile)(coord,
                                                           repeat<R1>(_));
}

template <class Tensor,
          class Tile,
          class Coord,
          class Projection,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto local_partition(Tensor&& tensor,
                                                Tile const& tile,
                                                Coord const& coord,
                                                Projection const& proj) {
  return local_partition(
      std::forward<Tensor>(tensor), dice(proj, tile), dice(proj, coord));
}

// Special case with Layout and Integral that extracts the coord first
// e.g. local_partition(tensor, ThrLayout, threadIdx.x)
template <class Tensor,
          class LShape,
          class LStride,
          class Index,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value&&
                              is_integral<Index>::value)>
CUTE_HOST_DEVICE auto local_partition(Tensor&& tensor,
                                      Layout<LShape, LStride> const& tile,
                                      Index const& index) {
  return local_partition(std::forward<Tensor>(tensor),
                         product_each(shape(tile)),
                         tile.get_flat_coord(index));
}

// Special case with Layout and Integral that extracts the coord first
// e.g. local_partition(tensor, ThrLayout, threadIdx.x, Step<_1,X,_1>{})
template <class Tensor,
          class LShape,
          class LStride,
          class Index,
          class Projection,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value&&
                              is_integral<Index>::value)>
CUTE_HOST_DEVICE auto local_partition(Tensor&& tensor,
                                      Layout<LShape, LStride> const& tile,
                                      Index const& index,
                                      Projection const& proj) {
  return local_partition(std::forward<Tensor>(tensor),
                         dice(proj, product_each(shape(tile))),
                         dice(proj, tile).get_flat_coord(index));
}

template <class Tensor,
          class Tile,
          class Coord,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE constexpr auto local_tile(Tensor&& tensor,
                                           Tile const& tile,
                                           Coord const& coord) {
  constexpr int R0 = decltype(rank(tile))::value;
  constexpr int R1 = decltype(rank(tensor))::value;

  // Split the modes of tensor according to the modes of tile
  // zipped_divide returns something like ((VEC_A,VEC_B,...),(a,b,...))

  // The padded_coord is the coord into the second mode, flatten the rest
  return zipped_divide(std::forward<Tensor>(tensor), tile)(
      repeat<R0>(_), append<R1>(coord, _));
}

template <class Tensor,
          class Tile,
          class Coord,
          class Proj,
          __CUTE_REQUIRES(is_tensor<remove_cvref_t<Tensor>>::value)>
CUTE_HOST_DEVICE auto local_tile(Tensor&& tensor,
                                 Tile const& tile,
                                 Coord const& coord,
                                 Proj const& proj) {
  return local_tile(
      std::forward<Tensor>(tensor), dice(proj, tile), dice(proj, coord));
}

//
// Display utilities
//

template <class Engine, class Layout>
CUTE_HOST_DEVICE void print_tensor(Tensor<Engine, Layout> const& tensor) {
  auto format = get_format(tensor(0));
  using type = typename decltype(format)::type;

  if constexpr (Layout::rank == 1) {
    for (int m = 0; m < size(tensor); ++m) {
      printf(format.format, format.digits, type(tensor(m)));
      printf("\n");
    }
  } else if constexpr (Layout::rank == 2) {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        printf(format.format, format.digits, type(tensor(m, n)));
      }
      printf("\n");
    }
  } else if constexpr (Layout::rank == 3) {
    print_tensor(tensor(_, _, 0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < format.digits * size<1>(tensor); ++i) {
        print("-");
      }
      print("\n");
      print_tensor(tensor(_, _, k));
    }
  } else if constexpr (Layout::rank == 4) {
    print_tensor(tensor(_, _, _, 0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < format.digits * size<1>(tensor); ++i) {
        print("=");
      }
      print("\n");
      print_tensor(tensor(_, _, _, p));
    }
  }
}

template <class Engine, class Layout>
CUTE_HOST_DEVICE void print(Tensor<Engine, Layout> const& tensor) {
  print(tensor.layout());
  print("\n");
  print_tensor(tensor);
}

template <class Engine, class Layout>
CUTE_HOST std::ostream& print_tensor_os(std::ostream& os,
                                        Tensor<Engine, Layout> const& tensor) {
  int digits = 9;

  if constexpr (Layout::rank == 1) {
    for (int m = 0; m < size(tensor); ++m) {
      os << std::setw(digits) << tensor(m) << std::endl;
    }
  } else if constexpr (Layout::rank == 2) {
    for (int m = 0; m < size<0>(tensor); ++m) {
      for (int n = 0; n < size<1>(tensor); ++n) {
        os << std::setw(digits) << tensor(m, n);
      }
      os << std::endl;
    }
  } else if constexpr (Layout::rank == 3) {
    print_tensor_os(os, tensor(_, _, 0));
    for (int k = 1; k < size<2>(tensor); ++k) {
      for (int i = 0; i < digits * size<1>(tensor); ++i) {
        os << "-";
      }
      os << std::endl;
      print_tensor_os(os, tensor(_, _, k));
    }
  } else if constexpr (Layout::rank == 4) {
    print_tensor_os(os, tensor(_, _, _, 0));
    for (int p = 1; p < size<3>(tensor); ++p) {
      for (int i = 0; i < digits * size<1>(tensor); ++i) {
        os << "=";
      }
      os << std::endl;
      print_tensor_os(os, tensor(_, _, _, p));
    }
  }

  return os;
}

template <class Engine, class Layout>
CUTE_HOST std::ostream& operator<<(std::ostream& os,
                                   Tensor<Engine, Layout> const& tensor) {
  os << tensor.layout() << std::endl;
  return print_tensor_os(os, tensor);
}

}  // end namespace cute

//
// Extended Engines
//

#include <cute/swizzle_ptr.hpp>

//
// Tensor Algorithms
//

#include <cute/algorithm/axpby.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/fill.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/tensor_algorithms.hpp>
