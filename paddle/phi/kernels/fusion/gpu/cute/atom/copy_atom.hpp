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

#include <type_traits>

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/tensor.hpp>

namespace cute {

// Generic copy_unpack for any Copy_Traits
template <class Operation,
          class... Args,
          class TS,
          class SLayout,
          class TD,
          class DLayout>
CUTE_HOST_DEVICE constexpr void copy_unpack(
    Copy_Traits<Operation, Args...> const&,
    Tensor<TS, SLayout> const& src,
    Tensor<TD, DLayout>& dst) {
  // Specializations can generalize on these checks
  // static_assert(is_smem<TS>::value, "Expected smem for this
  // Copy_Traits<Operation>"); static_assert(is_rmem<TD>::value, "Expected rmem
  // for this Copy_Traits<Operation>");

  using RegistersSrc = typename Operation::SRegisters;
  using RegistersDst = typename Operation::DRegisters;
  using RegTypeSrc = typename std::remove_extent<RegistersSrc>::type;
  using RegTypeDst = typename std::remove_extent<RegistersDst>::type;
  constexpr int RegNumSrc = std::extent<RegistersSrc>::value;
  constexpr int RegNumDst = std::extent<RegistersDst>::value;

  Tensor rS = recast<RegTypeSrc>(src);
  Tensor rD = recast<RegTypeDst>(dst);

  CUTE_STATIC_ASSERT_V(
      size(rS) == Int<RegNumSrc>{},
      "In CopyAtom, src layout doesn't vectorize into registers. This src "
      "layout is incompatible with this tiled copy.");
  CUTE_STATIC_ASSERT_V(
      size(rD) == Int<RegNumDst>{},
      "In CopyAtom, dst layout doesn't vectorize into registers. This dst "
      "layout is incompatible with this tiled copy.");

  detail::explode(Operation::copy,
                  rS,
                  make_int_sequence<RegNumSrc>{},
                  rD,
                  make_int_sequence<RegNumDst>{});
}

template <class... Args>
struct Copy_Atom;

template <class CopyOperation, class T>
struct Copy_Atom<CopyOperation, T> : Copy_Atom<Copy_Traits<CopyOperation>, T> {
};

template <class... Args, class T>
struct Copy_Atom<Copy_Traits<Args...>, T> : Copy_Traits<Args...> {
  using Traits = Copy_Traits<Args...>;

  // Bit and Thr layouts from the Copy_Traits
  using ThrID = typename Traits::ThrID;
  using BitLayoutSrc = typename Traits::SrcLayout;
  using BitLayoutDst = typename Traits::DstLayout;
  using BitLayoutRef = typename Traits::RefLayout;

  using ValType = T;

  using ValLayoutSrc =
      decltype(upcast<sizeof_bits<ValType>::value>(BitLayoutSrc{}));
  using ValLayoutDst =
      decltype(upcast<sizeof_bits<ValType>::value>(BitLayoutDst{}));
  using ValLayoutRef =
      decltype(upcast<sizeof_bits<ValType>::value>(BitLayoutRef{}));

  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutSrc{}) == size(ThrID{}),
                       "CopyOperation is not valid for Src of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutDst{}) == size(ThrID{}),
                       "CopyOperation is not valid for Dst of ValType.");
  CUTE_STATIC_ASSERT_V(size<0>(ValLayoutRef{}) == size(ThrID{}),
                       "CopyOperation is not valid for Ref of ValType.");

  static constexpr int NumValSrc = size<1>(ValLayoutSrc{});
  static constexpr int NumValDst = size<1>(ValLayoutDst{});

  // Additional Trait parameters/transformations
  template <class... TraitsArgs>
  CUTE_HOST_DEVICE auto with(TraitsArgs&&... args) const {
    auto traits = Traits::with(std::forward<TraitsArgs>(args)...);
    return Copy_Atom<decltype(traits), T>{traits};
  }

  // Print thread and data layouts for debugging
  CUTE_HOST_DEVICE static void print_all() {
    print("ThrID:        ");
    print(ThrID{});
    print("\n");
    print("BitLayoutSrc: ");
    print(BitLayoutSrc{});
    print("\n");
    print("BitLayoutDst: ");
    print(BitLayoutDst{});
    print("\n");
    print("BitLayoutRef: ");
    print(BitLayoutRef{});
    print("\n");
    print("ValLayoutSrc: ");
    print(ValLayoutSrc{});
    print("\n");
    print("ValLayoutDst: ");
    print(ValLayoutDst{});
    print("\n");
    print("ValLayoutRef: ");
    print(ValLayoutRef{});
    print("\n");
    print("ValueType:    %db", sizeof_bits<ValType>::value);
    print("\n");
  }

  //
  // Tensor call interfaces
  //

  // Cast, check, and call
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE void call(Tensor<TS, SLayout> const& src,
                             Tensor<TD, DLayout>& dst) const {
    static_assert(SLayout::rank == 1, "Expected rank-1 src tensor");
    static_assert(DLayout::rank == 1, "Expected rank-1 dst tensor");

    if constexpr (is_constant<NumValSrc, decltype(size(src))>::value ||
                  is_constant<NumValDst, decltype(size(dst))>::value) {
      // Dispatch to unpack for instruction
      return copy_unpack(*this, src, dst);
    } else {
      // Recurse if needed by peeling the tensor mode
      return copy(*this, tensor<0>(src), tensor<0>(dst));
    }
  }

  // Accept mutable temporaries
  template <class SEngine, class SLayout, class DEngine, class DLayout>
  CUTE_HOST_DEVICE void call(Tensor<SEngine, SLayout> const& src,
                             Tensor<DEngine, DLayout>&& dst) const {
    return call(src, dst);
  }
};

//
// A tiling of copy atoms
//

template <class Copy_Atom,
          class LayoutCopy_TV,  // (tid,vid) -> coord   [Need not be 2D...]
          class ShapeTile_MN>   // coord space
struct TiledCopy : Copy_Atom {
  // Layout information from the CopyAtom
  using AtomThrID = typename Copy_Atom::ThrID;  // thrid -> thr_idx
  using AtomLayoutSrc =
      typename Copy_Atom::ValLayoutSrc;  // (thr,val) -> offset
  using AtomLayoutDst =
      typename Copy_Atom::ValLayoutDst;  // (thr,val) -> offset
  using AtomLayoutRef =
      typename Copy_Atom::ValLayoutRef;  // (thr,val) -> offset

  using AtomNumThr = decltype(size<0>(AtomLayoutRef{}));
  using AtomNumVal = decltype(size<1>(AtomLayoutRef{}));

  // Layout information for the TiledCopy
  using Tiler_MN = ShapeTile_MN;
  using TiledShape_MN = decltype(shape(ShapeTile_MN{}));
  using TiledLayout_TV = LayoutCopy_TV;
  using TiledNumThr = decltype(size<0>(TiledLayout_TV{}));
  using TiledNumVal = decltype(size<1>(TiledLayout_TV{}));

  CUTE_STATIC_ASSERT_V(TiledNumThr{} % AtomNumThr{} == Int<0>{},
                       "TiledCopy uses too few thrs for selected CopyAtom");
  CUTE_STATIC_ASSERT_V(TiledNumVal{} % AtomNumVal{} == Int<0>{},
                       "TiledCopy uses too few vals for selected CopyAtom");

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,ThrX),FrgV,(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a COPY_ATOM Src.
  //   ThrX:  The threads tiled across COPY_ATOMs Src.
  //   FrgV:  The values local to a COPY_ATOM Src.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class STensor>
  CUTE_HOST_DEVICE constexpr static auto tidfrg_S(STensor&& stensor) {
    return thrfrg(stensor,
                  right_inverse(AtomLayoutRef{}).compose(AtomLayoutSrc{}));
  }

  // Tile a tensor or a layout from shape
  //   (M,N,...)
  // to shape
  //   ((ThrV,ThrX),FrgV,(RestM,RestN,...))
  // where
  //   ThrV:  The threads local to a COPY_ATOM Dst.
  //   ThrX:  The threads tiled across COPY_ATOMs Dst.
  //   FrgV:  The values local to a COPY_ATOM Dst.
  //   RestM: The values tiled in M.
  //   RestN: The values tiled in N.
  template <class DTensor>
  CUTE_HOST_DEVICE constexpr static auto tidfrg_D(DTensor&& dtensor) {
    return thrfrg(dtensor,
                  right_inverse(AtomLayoutRef{}).compose(AtomLayoutDst{}));
  }

  template <class Tensor, class Ref2TrgLayout>
  CUTE_HOST_DEVICE constexpr static auto thrfrg(Tensor&& tensor,
                                                Ref2TrgLayout const& ref2trg) {
    constexpr int R = remove_cvref_t<Tensor>::rank;
    static_assert(R >= rank_v<TiledShape_MN>,
                  "Rank of tensor to be partitioned too small.");
    // Generalize the dimension checks for arbitrary rank
    // CUTE_STATIC_ASSERT_V(size<0>(stensor) % size<0>(TiledShape_MNK{}) ==
    // Int<0>{}); CUTE_STATIC_ASSERT_V(size<1>(stensor) %
    // size<1>(TiledShape_MNK{}) == Int<0>{});

    // Take the thrs/vals that the atom is interested in
    // NOTE: Assumes the AtomNumThr are contiguous and identity within
    // TiledThrID
    auto atom_layout_TV =
        zipped_divide(TiledLayout_TV{}, make_shape(AtomNumThr{}, AtomNumVal{}));
    // ((atom_tid,atom_val),(rest_tid,rest_val)) -> (m,n)

    // Transform to the trg layout
    auto trg_layout_TV = atom_layout_TV.compose(ref2trg, _);
    // ((trg_tid,trg_val),(rest_tid,rest_val)) -> (m,n)

    // Transform the thrs mode from thrid to thr_idx
    // NOTE: Assumes the AtomNumThr are contiguous and identity within
    // TiledThrID
    auto thrval2mn = coalesce(zip(trg_layout_TV), Shape<_1, Shape<_1, _1>>{});
    // ((trg_tid,rest_tid),(trg_val,rest_val)) -> (m,n)

    /// ==================

    // Tile the tensor for TiledLayout
    auto t_tensor = zipped_divide(tensor, Tiler_MN{});
    // ((TileM,TileN,...),(RestM,RestN,...))

    // Transform the tile mode
    auto tv_tensor = t_tensor.compose(thrval2mn, _);
    // ((thrid,val),(RM,RN,...))

    // Unfold and return
    return tv_tensor(make_coord(_, _), _);
  }

  // retile_S and retile_D assume they are working with the reference layout --
  // they are the same
  template <class Tensor>
  CUTE_HOST_DEVICE constexpr static auto retile(Tensor&& tensor) {
    constexpr int R = remove_cvref_t<Tensor>::rank;
    // Assert that AtomLayoutSrc|Dst is identity so we can skip the Ref
    // transformation

    // Assume the first size<0>(tensor) elements are the first val_ids in
    // TiledLayout_TV. Then, we only need the shape+layout of those
    // size<0>(tensor) elements in TiledLayout_TV
    //   and that shape is what we gather from the other modes of tensor

    auto V = size<0>(tensor);

    auto frg_layout_mn = upcast<TiledNumThr{} * V>(
        right_inverse(TiledLayout_TV{}).with_shape(TiledShape_MN{}));
    // (m,n) -> v_idx -- The shape and order of the V inside of TiledLayout_TV

    auto frg_layout_v = zipped_divide(
        logical_product(make_layout(V), right_inverse(frg_layout_mn)),
        make_layout(AtomNumVal{}));
    // (atom_vals,rest_vals) -> (v,m,n)

    /// =======

    // Tile the tensor for TileFrg
    auto t_tensor =
        zipped_divide(tensor, prepend(product_each(shape(frg_layout_mn)), V));
    // ((TileV,TileM,TileN,...),(1,RestM,RestN,...))

    // Transform the tile mode
    auto v_tensor = t_tensor.compose(frg_layout_v, _);
    // ((atom_vals,rest_vals),(1,RM,RN,...))

    // Unfold and return
    return v_tensor(_, append<R>(Int<0>{}, _));
  }

  CUTE_HOST_DEVICE constexpr static auto get_layoutS_MN() {
    // (M,N) -> (M,N)
    auto ref_S = make_layout(TiledShape_MN{});
    // (thr_idx,val_idx) -> (M,N)
    auto layoutS_TV = tidfrg_S(ref_S);
    // (M,K) -> (thr_idx,val_idx)
    auto layoutS_MK = right_inverse(layoutS_TV).with_shape(shape(ref_S));

    // athrid = (v,m,k) -> thr_idx
    auto thrID_S = make_layout(size<0>(TiledLayout_TV{}));

    return cute::make_tuple(layoutS_MK, thrID_S);
  }

  CUTE_HOST_DEVICE constexpr static auto get_layoutS_TV() {
    // (M,N) -> (M,N)
    auto ref_S = make_layout(TiledShape_MN{});
    // (thr_idx,val_idx) -> (M,N)
    return tidfrg_S(ref_S)(_, _, Int<0>{});
  }

  CUTE_HOST_DEVICE constexpr static auto get_layoutD_MN() {
    // (M,N) -> (M,N)
    auto ref_D = make_layout(TiledShape_MN{});
    // (thr_idx,val_idx) -> (M,N)
    auto layoutD_TV = tidfrg_D(ref_D);
    // (M,K) -> (thr_idx,val_idx)
    auto layoutD_MK = right_inverse(layoutD_TV).with_shape(shape(ref_D));

    // athrid = (v,m,k) -> thr_idx
    auto thrID_D = make_layout(size<0>(TiledLayout_TV{}));

    return cute::make_tuple(layoutD_MK, thrID_D);
  }

  CUTE_HOST_DEVICE constexpr static auto get_layoutD_TV() {
    // (M,N) -> (M,N)
    auto ref_D = make_layout(TiledShape_MN{});
    // (thr_idx,val_idx) -> (M,N)
    return tidfrg_D(ref_D)(_, _, Int<0>{});
  }

  template <class ThrIdx>
  struct ThrCopy : Copy_Atom {
    ThrIdx thr_idx_;

    CUTE_HOST_DEVICE
    ThrCopy(ThrIdx const& thr_idx) : thr_idx_(thr_idx) {}

    template <class STensor>
    CUTE_HOST_DEVICE auto partition_S(STensor&& stensor) {
      // static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) ==
      // sizeof(typename Copy_Atom::ValType),
      //               "Expected ValType for tiling SrcTensor.");
      auto thr_tensor = make_tensor(std::forward<STensor>(stensor).data(),
                                    tidfrg_S(stensor.layout()));
      return thr_tensor(thr_idx_, _, repeat<rank_v<STensor>>(_));
    }

    template <class DTensor>
    CUTE_HOST_DEVICE auto partition_D(DTensor&& dtensor) {
      // static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) ==
      // sizeof(typename Copy_Atom::ValType),
      //               "Expected ValType for tiling DstTensor.");
      auto thr_tensor = make_tensor(std::forward<DTensor>(dtensor).data(),
                                    tidfrg_D(dtensor.layout()));
      return thr_tensor(thr_idx_, _, repeat<rank_v<DTensor>>(_));
    }

    template <class STensor>
    CUTE_HOST_DEVICE static auto retile_S(STensor&& stensor) {
      static_assert(sizeof(typename remove_cvref_t<STensor>::value_type) ==
                        sizeof(typename Copy_Atom::ValType),
                    "Expected ValType for tiling SrcTensor.");
      return make_tensor(std::forward<STensor>(stensor).data(),
                         TiledCopy::retile(stensor.layout()));
    }

    template <class DTensor>
    CUTE_HOST_DEVICE static auto retile_D(DTensor&& dtensor) {
      static_assert(sizeof(typename remove_cvref_t<DTensor>::value_type) ==
                        sizeof(typename Copy_Atom::ValType),
                    "Expected ValType for tiling DstTensor.");
      return make_tensor(std::forward<DTensor>(dtensor).data(),
                         TiledCopy::retile(dtensor.layout()));
    }
  };

  template <class ThrIdx, __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE static auto get_slice(ThrIdx const& thr_idx) {
    return ThrCopy<ThrIdx>(thr_idx);
  }

  template <class ThrIdx, __CUTE_REQUIRES(is_integral<ThrIdx>::value)>
  CUTE_HOST_DEVICE static auto get_thread_slice(ThrIdx const& thr_idx) {
    return get_slice(thr_idx);
  }
};

template <class... Args, class LayoutCopy_TV, class... TLayout>
CUTE_HOST_DEVICE auto make_tiled_copy_impl(Copy_Atom<Args...> const& atom,
                                           LayoutCopy_TV const&,
                                           Tile<TLayout...> const&) {
  return TiledCopy<Copy_Atom<Args...>, LayoutCopy_TV, Tile<TLayout...>>{atom};
}

//
// These tile the Copy_Atom as a whole
//

template <class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_A(Copy_Atom<Args...> const& copy_atom,
                                        TiledMMA const& tiled_mma) {
  using MNK = typename TiledMMA::TiledShape_MNK;
  return make_tiled_copy_impl(copy_atom,
                              tiled_mma.get_layoutA_TV(),
                              make_shape(size<0>(MNK{}), size<2>(MNK{})));
}

template <class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_B(Copy_Atom<Args...> const& copy_atom,
                                        TiledMMA const& tiled_mma) {
  using MNK = typename TiledMMA::TiledShape_MNK;
  return make_tiled_copy_impl(copy_atom,
                              tiled_mma.get_layoutB_TV(),
                              make_shape(size<1>(MNK{}), size<2>(MNK{})));
}

template <class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto make_tiled_copy_C(Copy_Atom<Args...> const& copy_atom,
                                        TiledMMA const& tiled_mma) {
  using MNK = typename TiledMMA::TiledShape_MNK;
  return make_tiled_copy_impl(copy_atom,
                              tiled_mma.get_layoutC_TV(),
                              make_shape(size<0>(MNK{}), size<1>(MNK{})));
}

template <class... Args, class ThrLayout, class ValLayout = Layout<_1>>
CUTE_HOST_DEVICE auto make_tiled_copy(
    Copy_Atom<Args...> const& copy_atom,
    ThrLayout const& thr_layout = {},  // (m,n) -> thr_idx
    ValLayout const& val_layout = {}) {
  constexpr int R = cute::max(rank_v<ThrLayout>, rank_v<ValLayout>);

  auto thr_layout_mn = append<R>(thr_layout, Layout<_1>{});
  auto val_layout_mn = append<R>(val_layout, Layout<_1>{});

  // Take the raked_products to compute the Layout_MN
  auto layout_mn = raked_product(thr_layout_mn, val_layout_mn);
  auto layout_tv = right_inverse(layout_mn).with_shape(
      make_shape(size(thr_layout), size(val_layout)));

  // print("thr_layout: "); print(thr_layout_mn); print("\n");
  // print("val_layout: "); print(val_layout_mn); print("\n");
  // print("layout_mn : "); print(layout_mn);     print("\n");
  // print("layout_tv : "); print(layout_tv);     print("\n");

  return make_tiled_copy_impl(
      copy_atom, layout_tv, product_each(shape(layout_mn)));
}

// Make a TiledCopy out of the copy_atom that matches the Src-Layout of
// tiled_copy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE auto make_tiled_copy_S(Copy_Atom<Args...> const& copy_atom,
                                        TiledCopy const& tiled_copy) {
  return make_tiled_copy_impl(
      copy_atom, tiled_copy.get_layoutS_TV(), typename TiledCopy::Tiler_MN{});
}

// Make a TiledCopy out of the copy_atom that matches the Dst-Layout of
// tiled_copy
template <class... Args, class TiledCopy>
CUTE_HOST_DEVICE auto make_tiled_copy_D(Copy_Atom<Args...> const& copy_atom,
                                        TiledCopy const& tiled_copy) {
  return make_tiled_copy_impl(
      copy_atom, tiled_copy.get_layoutD_TV(), typename TiledCopy::Tiler_MN{});
}

//
// Size
//

// The logical size of a TileCopy
template <int... I, class... Args>
CUTE_HOST_DEVICE constexpr auto tile_size(TiledCopy<Args...> const&) {
  return size<I...>(typename TiledCopy<Args...>::TiledShape_MN{});
}

// The number of threads involved in a TiledCopy
template <class... Args>
CUTE_HOST_DEVICE constexpr auto size(TiledCopy<Args...> const&) {
  return typename TiledCopy<Args...>::TiledNumThr{};
}

//
// Display utilities
//

template <class... Args>
CUTE_HOST_DEVICE auto print_latex(TiledCopy<Args...> const& copy) {
  auto [layoutS_MN, thrID_S] = copy.get_layoutS_MN();
  auto [layoutD_MN, thrID_D] = copy.get_layoutD_MN();

  print_latex_copy(layoutS_MN, thrID_S, layoutD_MN, thrID_D);
}

// MNK Copy Layout to Latex TIKZ -- 8-value color coded by thread
template <class LayoutS, class ThrIDS, class LayoutD, class ThrIDD>
CUTE_HOST_DEVICE void print_latex_copy(
    LayoutS const& S,
    ThrIDS const& TS,  // (m,n) -> (tid,vid)  and  tid -> thr_idx
    LayoutD const& D,
    ThrIDD const& TD)  // (m,n) -> (tid,vid)  and  tid -> thr_idx
{
  CUTE_STATIC_ASSERT_V(rank(S) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<2>{});

  assert(size<0>(S) == size<0>(D));
  assert(size<1>(S) == size<1>(D));

  char const* latex_header =
      "\\documentclass{standalone}\n"
      "\\usepackage{tikz}\n"
      "\\usetikzlibrary{external}\n"
      "\\tikzexternalize\n"
      "\\begin{document}\n"
      "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},box/"
      ".style={rectangle,draw=black,thick,minimum size=1cm,anchor=center}]\n\n";
  char const* latex_footer =
      "\\end{tikzpicture}\n"
      "\\end{document}\n";

  char const* color_map[8] = {
      "{rgb,255:red,175;green,175;blue,255}",
      "{rgb,255:red,175;green,255;blue,175}",
      "{rgb,255:red,255;green,255;blue,175}",
      "{rgb,255:red,255;green,175;blue,175}",
      "{rgb,255:red,210;green,210;blue,255}",
      "{rgb,255:red,210;green,255;blue,210}",
      "{rgb,255:red,255;green,255;blue,210}",
      "{rgb,255:red,255;green,210;blue,210}",
  };

  // Header
  printf("%% LayoutS: ");
  print(S);
  printf("\n");
  printf("%% ThrIDS : ");
  print(TS);
  printf("\n");
  printf("%% LayoutD: ");
  print(D);
  printf("\n");
  printf("%% ThrIDD : ");
  print(TD);
  printf("\n\n");

  printf(latex_header);

  // S starting at 0,0
  for (int i = 0; i < size<0>(S); ++i) {
    for (int j = 0; j < size<1>(S); ++j) {
      int thrid = S(i, j) % size(TS);
      int val_idx = S(i, j) / size(TS);
      int thr_idx = TS(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i,
             j,
             thr_idx,
             val_idx);
    }
  }

  // D starting at 0,size<1>(S)+3
  for (int i = 0; i < size<0>(D); ++i) {
    for (int j = 0; j < size<1>(D); ++j) {
      int thrid = D(i, j) % size(TD);
      int val_idx = D(i, j) / size(TD);
      int thr_idx = TD(thrid);

      printf("\\node[box,fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color_map[thr_idx % 8],
             i,
             j + size<1>(S) + 3,
             thr_idx,
             val_idx);
    }
  }

  // S Labels
  for (int i = 0, j = -1; i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int j = 0, i = -1; j < size<1>(S); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }
  // D Labels
  for (int i = 0, j = size<1>(D); i < size<0>(S); ++i) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n",
           i,
           j + size<1>(S) + 3,
           i);
  }
  for (int j = 0, i = -1; j < size<1>(D); ++j) {
    printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n",
           i,
           j + size<1>(S) + 3,
           j);
  }

  // Footer
  printf(latex_footer);
}

}  // end namespace cute

////////////////////////////////////////////////////////////////////////////////////////////////////

#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_traits_sm75.hpp>
#include <cute/atom/copy_traits_sm80.hpp>
#include <cute/atom/copy_traits_sm90.hpp>
// Config
#if (__CUDACC_VER_MAJOR__ >= 12)
#define CUTE_COPY_ATOM_TMA_SM90_ENABLED
#endif

#if defined(CUTE_COPY_ATOM_TMA_SM90_ENABLED)
#include <cute/atom/copy_traits_sm90_tma.hpp>
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
