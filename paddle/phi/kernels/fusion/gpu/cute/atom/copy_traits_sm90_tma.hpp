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

#include <cuda.h>

#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/tensor.hpp>

namespace cute {

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD ///////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_OP : SM90_TMA_LOAD {};

// The executable SM90_TMA_LOAD with tma_desc and tma_mbar
template <class NumBits>
struct Copy_Traits<SM90_TMA_LOAD_OP, NumBits> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBits>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBits>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor const& tma_desc_;
  uint64_t& tma_load_mbar_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr void copy_unpack_(void const* const dst_ptr,
                                               Coord const& src_coord,
                                               seq<Is...>) const {
#if 0
    print("THR (%d,%d,%d) BLK (%d,%d,%d)\n",
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z);
    print("  TMA Coord "); print(src_coord); print("\n");
    print("  TMA Shape "); print(make_tuple(uint64_t(tma_desc_.size0_),
                                            uint64_t(tma_desc_.size1_),
                                            uint64_t(tma_desc_.size2_),
                                            uint64_t(tma_desc_.size3_))); print("\n");
#endif

    SM90_TMA_LOAD::copy(
        &tma_desc_, tma_load_mbar_, dst_ptr, get<Is>(src_coord)...);
  }

  // This is the copy_unpack dispatch for this Copy_Traits
  // Src needs to be a gmem tensor with TmaCoordIterator .data()
  // Dst needs to be a smem tensor
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits,
      Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) {
    // static_assert(is_gmem<TS>::value, "Expected gmem src for SM90_TMA_LOAD");
    // // TMA spoofed src tensor
    static_assert(is_smem<TD>::value, "Expected smem dst for SM90_TMA_LOAD");

    traits.copy_unpack_(dst.data().get(),
                        src.data().coord_,
                        tuple_seq<decltype(src.data().coord_)>{});
  }
};

// The non-executable SM90_TMA_LOAD with tma_desc and no tma_mbar
// Use .with(tma_mbar) to construct an executable version
template <class NumBits, class GmemStrides>
struct Copy_Traits<SM90_TMA_LOAD, NumBits, GmemStrides> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBits>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBits>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD arguments
  TmaDescriptor tma_desc_;
  GmemStrides g_stride_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr TmaDescriptor const* get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD with tma_mbar
  CUTE_HOST_DEVICE constexpr Copy_Traits<SM90_TMA_LOAD_OP, NumBits> with(
      uint64_t& tma_mbar, uint16_t const& multicast_mask = 0) const {
    // We accept multicast_mask here to keep the API for both atoms consistent
    // assert(multicast_mask == 0);
    (void)multicast_mask;
    return {tma_desc_, tma_mbar};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(g_stride_)>::value);
    constexpr int tma_rank =
        decltype(cute::min(rank(flatten(g_stride_)), Int<5>{}))::value;
    return make_tensor(ArithmeticTupleIterator(
                           as_arithmetic_tuple(repeat<tma_rank>(Int<0>{}))),
                       g_shape,
                       g_stride_);
  }

  // Don't try to execute a copy with SM90_TMA_LOAD before calling .with()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits,
      Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) = delete;
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_LOAD_MULTICAST /////////////////////////////
//////////////////////////////////////////////////////////////////////////////

struct SM90_TMA_LOAD_MULTICAST_OP : SM90_TMA_LOAD_MULTICAST {};

template <class NumBits>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBits> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBits>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBits>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  TmaDescriptor const& tma_desc_;
  uint64_t& tma_load_mbar_;
  uint16_t const& multicast_mask_;

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr void copy_unpack_(void const* const dst_ptr,
                                               Coord const& src_coord,
                                               seq<Is...>) const {
#if 0
    print("THR (%d,%d,%d) BLK (%d,%d,%d)\n",
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z);
    print("  TMA Coord "); print(src_coord); print("\n");
    print("  TMA Shape "); print(make_tuple(uint64_t(tma_desc_.size0_),
                                            uint64_t(tma_desc_.size1_),
                                            uint64_t(tma_desc_.size2_),
                                            uint64_t(tma_desc_.size3_))); print("\n");
#endif

    SM90_TMA_LOAD_MULTICAST::copy(&tma_desc_,
                                  tma_load_mbar_,
                                  multicast_mask_,
                                  dst_ptr,
                                  get<Is>(src_coord)...);
  }

  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits,
      Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) {
    // static_assert(is_gmem<TS>::value, "Expected gmem src for SM90_TMA_LOAD");
    // // TMA spoofed src tensor
    static_assert(is_smem<TD>::value,
                  "Expected smem dst for SM90_TMA_LOAD_MULTICAST");

    traits.copy_unpack_(dst.data().get(),
                        src.data().coord_,
                        tuple_seq<decltype(src.data().coord_)>{});
  }
};

template <class NumBits, class GmemStrides>
struct Copy_Traits<SM90_TMA_LOAD_MULTICAST, NumBits, GmemStrides> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBits>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBits>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_LOAD_MULTICAST arguments
  TmaDescriptor tma_desc_;
  GmemStrides g_stride_;

  // Return TmaDescriptor/TensorMap
  CUTE_HOST_DEVICE constexpr TmaDescriptor const* get_tma_descriptor() const {
    return &tma_desc_;
  }

  // Construct an executable SM90_TMA_LOAD_MULTICAST with tma_mbar
  CUTE_HOST_DEVICE constexpr Copy_Traits<SM90_TMA_LOAD_MULTICAST_OP, NumBits>
  with(uint64_t& tma_load_mbar, uint16_t const& multicast_mask) const {
    return {tma_desc_, tma_load_mbar, multicast_mask};
  }

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(g_stride_)>::value);
    constexpr int tma_rank =
        decltype(cute::min(rank(flatten(g_stride_)), Int<5>{}))::value;
    return make_tensor(ArithmeticTupleIterator(
                           as_arithmetic_tuple(repeat<tma_rank>(Int<0>{}))),
                       g_shape,
                       g_stride_);
  }

  // Don't try to execute a copy with SM90_TMA_LOAD_MULTICAST before calling
  // .with()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits,
      Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) = delete;
};

//////////////////////////////////////////////////////////////////////////////
///////////////////////////// TMA_STORE //////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

// The executable SM90_TMA_STORE with tma_desc
template <class NumBits, class GmemStrides>
struct Copy_Traits<SM90_TMA_STORE, NumBits, GmemStrides> {
  using ThrID = Layout<_1>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_1, NumBits>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_1, NumBits>>;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;

  // SM90_TMA_STORE arguments
  TmaDescriptor tma_desc_;
  GmemStrides g_stride_;

  // Generate the TMA coord tensor
  template <class GShape>
  CUTE_HOST_DEVICE constexpr auto get_tma_tensor(GShape const& g_shape) const {
    static_assert(is_congruent<decltype(g_shape), decltype(g_stride_)>::value);
    constexpr int tma_rank =
        decltype(cute::min(rank(flatten(g_stride_)), Int<5>{}))::value;
    return make_tensor(ArithmeticTupleIterator(
                           as_arithmetic_tuple(repeat<tma_rank>(Int<0>{}))),
                       g_shape,
                       g_stride_);
  }

  template <class Coord, int... Is>
  CUTE_HOST_DEVICE constexpr void copy_unpack_(void const* const src_ptr,
                                               Coord const& dst_coord,
                                               seq<Is...>) const {
#if 0
    print("THR (%d,%d,%d) BLK (%d,%d,%d)\n",
          threadIdx.x, threadIdx.y, threadIdx.z,
          blockIdx.x, blockIdx.y, blockIdx.z);
    print("  TMA Coord "); print(dst_coord); print("\n");
    print("  TMA Shape "); print(make_tuple(uint64_t(tma_desc_.size0_),
                                            uint64_t(tma_desc_.size1_),
                                            uint64_t(tma_desc_.size2_),
                                            uint64_t(tma_desc_.size3_))); print("\n");
#endif

    SM90_TMA_STORE::copy(&tma_desc_, src_ptr, get<Is>(dst_coord)...);
  }

  // This is the copy_unpack dispatch for this Copy_Traits
  // Src needs to be a smem tensor
  // Dst needs to be a gmem tensor with TmaCoordIterator .data()
  template <class TS, class SLayout, class TD, class DLayout>
  CUTE_HOST_DEVICE friend constexpr void copy_unpack(
      Copy_Traits const& traits,
      Tensor<TS, SLayout> const& src,
      Tensor<TD, DLayout>& dst) {
    static_assert(is_smem<TS>::value, "Expected smem src for SM90_TMA_STORE");
    // static_assert(is_gmem<TD>::value, "Expected gmem dst for
    // SM90_TMA_STORE");  // TMA spoofed src tensor

    traits.copy_unpack_(src.data().get(),
                        dst.data().coord_,
                        tuple_seq<decltype(dst.data().coord_)>{});
  }
};

//
// MAKE_TMA_COPY and related
//

template <int B, int M, int S, class Offset, class SLayout>
TMA::SmemSwizzleBits get_tma_swizzle_bits(
    ComposedLayout<Swizzle<B, M, S>, Offset, SLayout>) {
  static_assert(M == 4, "Expected 128b=16B=(2^4)B base swizzle.");
  static_assert(S == 3, "Unsupported layout swizzle");

  switch (B) {
    default:
      static_assert(0 <= B && B <= 3,
                    "Expected B = 0,1,2, or 3. Unsupported layout swizzle.");
    case 3:
      return TMA::SmemSwizzleBits::B128;
    case 2:
      return TMA::SmemSwizzleBits::B64;
    case 1:
      return TMA::SmemSwizzleBits::B32;
    case 0:
      return TMA::SmemSwizzleBits::DISABLE;
  }
}

template <class Shape, class Stride>
TMA::SmemSwizzleBits get_tma_swizzle_bits(Layout<Shape, Stride>) {
  return TMA::SmemSwizzleBits::DISABLE;
}

template <int B, int M, int S, class Offset, class SLayout>
auto get_nonswizzle_layout(
    ComposedLayout<Swizzle<B, M, S>, Offset, SLayout> const& slayout) {
  return slayout.layout_fn();
}

template <class Shape, class Stride>
auto get_nonswizzle_layout(Layout<Shape, Stride> const& slayout) {
  return slayout;
}

/** Make a CuTe CTA-collective TiledCopy for a TMA operation.
 *
 * @param CopyOp The target copy operation: SM90_TMA_LOAD,
 SM90_TMA_LOAD_MULTICAST, SM90_TMA_STORE
 * @param gtensor The GMEM Tensor to be involved in the TMA.
 * @param slayout The SMEM Layout to be involved in the TMA.
 * @param cta_tile The CTA-local tile that each CTA will be tiling GMEM with.
 *                 This is often the blk_shape that is used to tile the GMEM for
 CTAs:
 *                   local_tile(gtensor, blk_shape, blk_coord) -> CTA-local tile
 of gtensor
 * @param cluster_size When using SM90_TMA_LOAD_MULTICAST, this can be a
 (static) power-of-2 <= 16
 *                   defining the multicast size (used to further partition the
 SMEM)
 *                 Else, static-1
 *
 * This code attempts to maximize the TMA box size. It does this by tracing
 * the SMEM "vector" -- the inverse of the smem layout -- to find the largest
 * contiguous array of smem that can be written to/from global memory given
 * the constraints that the TMA instruction imposes.
 *
 * This is accomplished by assigning "basis" strides to the GMEM to track which
 * modes of SMEM map to which modes of GMEM, then reorder the modes of GMEM
 according
 * to the SMEM vector, and then using those GMEM/SMEM modes to fill in the desc.
 *
 * Examples:
     using T = float;
     T* gptr = nullptr;

    {
    // Simple 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256), GenRowMajor{}); //
 K-Major GMEM auto slayout   = make_layout(make_shape(_64{}, _32{}),
 GenRowMajor{});    // K-Major SMEM auto tma = make_tma_copy(SM90_TMA_LOAD{},
 gtensor, slayout);
    }

    {
    // GMMA 2D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 256)); // MN-Major GMEM
    auto slayout   = tile_to_shape(GMMA::Layout_MN_SW128_Atom<T>{},
 make_shape(_128{},_64{})); // MN-Major Swizzled+Tiled 128x64 SMEM auto tma =
 make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout);
    }

    {
    // 3D
    Tensor gtensor = make_tensor(gptr, make_shape(1024, 32, 512),
 make_stride(64, Int<1>{}, 65536)); // GMEM auto slayout   =
 make_layout(make_shape(_16{}, _8{}, _2{}), make_stride(_16{}, _1{}, _8{})); //
 SMEM w/ same major-mode auto tma = make_tma_copy(SM90_TMA_LOAD{}, gtensor,
 slayout);
    }

    {
    // cuTENSOR 4D
    auto layout = make_shape(make_shape(32,40),make_shape(make_shape(8,8),656));
 // GMEM auto cta_tile    = make_shape(_128{},make_shape(_32{},_2{})); // GMEM
 Tiling:
                                                                                 //   Take 128-elem from m: m0 must divide 128,
                                                                                 //                         m-last may be predicated
                                                                                 //   Take 32-elem from k0, 2-elem from k1
    auto slayout = make_layout(cta_tile); // Col-Major SMEM auto tma =
 make_tma_copy(SM90_TMA_LOAD{}, gtensor, slayout, cta_tile, Int<1>{});
    }
 *
 * Check the TMA box size and desc:
    print("TMA Box size:  "); print(typename decltype(tma)::Tiler_MN{});
 print("\n"); print("TMA desc     : "); print(tma.tma_desc_); print("\n");
 *
 * Usage:
     Tensor mA = tma_a.get_tma_tensor(make_shape(M,N));        // (M,N) TMA
 coord tensor Tensor gA = local_tile(mA, cta_tile, cta_coord);          //
 (BLK_M,BLK_N) TMA coord tensor for this CTA Tensor sA =
 make_tensor(make_smem_ptr<T>(sptr), slayout); // (BLK_M,BLK_N) SMEM tensor

     auto cta_tma = tma.get_slice(cta_idx_in_cluster);         // Slice for
 multicast partitioning Tensor tAgA = cta_tma.partition_S(gA); // Partition for
 src Tensor tAsA = cta_tma.partition_D(sA);                    // Partition for
 dst

     copy(tma.with(barrier, mcast_mask), tAgA, tAsA);          // copy with
 supporting TMA params
 */
template <class CopyOp,
          class GEngine,
          class GLayout,
          class SLayout,
          class CTA_Tile,
          class Cluster_Size>
CUTE_HOST auto make_tma_copy(CopyOp,
                             Tensor<GEngine, GLayout> const& gtensor,
                             SLayout const& slayout,
                             CTA_Tile const& cta_tile,
                             Cluster_Size const& cluster_size) {
  static_assert((std::is_same<CopyOp, SM90_TMA_LOAD>::value &&
                 is_constant<1, Cluster_Size>::value) ||
                (std::is_same<CopyOp, SM90_TMA_LOAD_MULTICAST>::value) ||
                (std::is_same<CopyOp, SM90_TMA_STORE>::value &&
                 is_constant<1, Cluster_Size>::value));

  using T = typename Tensor<GEngine, GLayout>::value_type;

  //
  // TMA parameter checking
  //

  auto flat_glayout = flatten(gtensor.layout());

  CUTE_STATIC_ASSERT_V(
      rank(flatten(cta_tile)) <= Int<5>{},
      "CTA_Tile cannot have more than five modes, TMA arch restriction.");
  CUTE_STATIC_ASSERT_V(
      rank(flat_glayout) <= Int<5>{} || rank(flatten(cta_tile)) <= Int<4>{},
      "If GTensor has more than five modes, then CTA_Tile cannot have more "
      "than four modes. TMA multimode.");
  CUTE_STATIC_ASSERT_V(
      compatible(product_each(shape(slayout)), shape(cta_tile)),
      "CTA_Tile must be compatible with SLayout.");
  CUTE_STATIC_ASSERT_V(is_integral<Cluster_Size>{} &&
                           has_single_bit(cluster_size) &&
                           cluster_size <= Int<16>{},
                       "Expecting a pow2 integral Cluster_Size leq 16.");
  CUTE_STATIC_ASSERT_V(size(slayout) % cluster_size == Int<0>{},
                       "ClusterShape must divide domain size of slayout.");

  //
  // TMA slayout manipulation
  //

  auto tma_multimode = rank(flat_glayout) > Int<5>{};

  // Invert the smem to get the largest contiguous vector in the smem layout
  auto inv_smem_layout = right_inverse(get_nonswizzle_layout(slayout));
  // trunc_smem_idx -> trunc_smem_coord

  // Map from smem idx to a gmem mode
  auto sidx_to_gmode =
      flatten(composition(make_identity_layout(cta_tile), inv_smem_layout));

  // Truncate any incompatibilities
  auto smem_rank = find_if(stride(sidx_to_gmode), [](auto e) {
    [[maybe_unused]] auto v = basis_value(e);
    return not is_constant<1, decltype(v)>{};
  });
  static_assert(smem_rank > 0,
                "Could not find a common smem-gmem vectorization for TMA.");
  constexpr int smem_tma_rank =
      cute::min(int(smem_rank), (tma_multimode ? 4 : 5));

  // Keep only the static-1 basis modes into gmem
  auto sidx_to_gmode_cluster_trunc = take<0, smem_tma_rank>(sidx_to_gmode);
  // Keep only the portion each multicast CTA will be responsible for
  auto sidx_to_gmode_cta_trunc =
      composition(sidx_to_gmode_cluster_trunc,
                  shape_div(size(sidx_to_gmode_cluster_trunc), cluster_size));

  //
  // TMA gtensor manipulation
  //

  // Generate a TupleBasis for the gtensor
  auto flat_gbasis = make_basis_like(shape(flat_glayout));

  // Fold the flat_gbasis into the glayout
  auto glayout_basis = make_layout(
      shape(gtensor),
      stride(composition(
          make_layout(repeat_like(shape(flat_glayout), Int<2>{}), flat_gbasis),
          make_layout(repeat_like(shape(gtensor), Int<2>{})))));

  // Tile the modes of gtensor with cta_tile
  auto cta_glayout_basis = composition(glayout_basis, cta_tile);

  // Check that the cta_tile selects modes from gtensor properly
  for_each(flatten(stride(cta_glayout_basis)), [](auto d) {
    static_assert(is_constant<1, decltype(d.value())>::value,
                  "CTA_Tile does not faithfully partition the GMEM, it should "
                  "select the number of elements from each mode of glayout.");
  });

  // Tile the modes of gtensor again with the truncated cta_tile o
  // inv_smem_layout
  auto tma_layout_cta_trunc =
      flatten(composition(glayout_basis, sidx_to_gmode_cta_trunc));

  // Append any missing basis on the end as size-1 modes b/c they got truncated
  auto missing_basis =
      fold(stride(tma_layout_cta_trunc), flat_gbasis, [](auto init, auto e) {
        auto k = find(init, e);
        return remove<k>(init);
      });

  // The appended map from truncated smem codomain to gmem mode: trunc_smem_idx
  // -> gmem_mode
  auto tma_layout_cta = flatten(make_layout(
      tma_layout_cta_trunc,
      make_layout(repeat<rank(missing_basis)>(Int<1>{}), missing_basis)));

#if 0
  print("g_layout      : "); print(gtensor.layout()); print("\n");
  print("s_layout      : "); print(slayout); print("\n");
  print("cta_tile      : "); print(cta_tile); print("\n");
  print("cluster_size      : "); print(cluster_size); print("\n");
  print("flat_gbasis   : "); print(flat_gbasis); print("\n");
  print("cta_glayout   : "); print(cta_glayout_basis); print("\n");
  print("inv_smem      : "); print(inv_smem_layout); print("\n");
  print("sidx_to_gmode : "); print(sidx_to_gmode); print("\n");
  print("missing_b     : "); print(missing_basis); print("\n");
  print("tma_layout_cta: "); print(tma_layout_cta); print("\n");
#endif

  //
  // TMA gmem desc info
  //

  constexpr int TmaRANK = cute::min(rank(flat_glayout), 5);
  void* gmem_address = (void*)gtensor.data();

  cute::array<cuuint64_t, 5> gmem_prob_shape = {1, 1, 1, 1, 1};
  cute::array<cuuint64_t, 5> gmem_prob_stride = {0, 0, 0, 0, 0};
  for_each(make_seq<rank(tma_layout_cta)>{}, [&](auto i) {
    // NOTE : WAR g++-7.3.5, let it deduce e rather than fuse with below
    auto e = stride<i>(tma_layout_cta);
    constexpr int j = decltype(e.mode())::value;
    constexpr int tma_i = i < 5 ? i : 4;

    // Problem stride
    uint64_t stride_j = stride<j>(flat_glayout) * sizeof(T);
    uint64_t old_stride = gmem_prob_stride[tma_i];
    gmem_prob_stride[tma_i] = gcd(gmem_prob_stride[tma_i], stride_j);

    // Problem shape
    uint64_t shape_j = shape<j>(flat_glayout);
    if (gmem_prob_stride[tma_i] != 0) {
      // We're "resetting" this TMA mode and using it as a "multimode"
      // Recurrence: g_shape = (s_i - 1) * (d_i / gcd_j d_j) + 1
      gmem_prob_shape[tma_i] =
          (gmem_prob_shape[tma_i] - 1) *
              (old_stride / gmem_prob_stride[tma_i]) +
          (shape_j - 1) * (stride_j / gmem_prob_stride[tma_i]) + 1;
    } else {
      gmem_prob_shape[tma_i] = shape_j;
    }
  });

  assert((reinterpret_cast<uint64_t>(gmem_address) & 0b1111) ==
         0);  // Address must be 16B-aligned

  assert(gmem_prob_shape[0] >= (uint64_t(1)));        // Size must be min 1
  assert(gmem_prob_shape[0] <= (uint64_t(1) << 32));  // Size must be max 2^32
  assert(gmem_prob_shape[1] >= (uint64_t(1)));        // Size must be min 1
  assert(gmem_prob_shape[1] <= (uint64_t(1) << 32));  // Size must be max 2^32
  assert(gmem_prob_shape[2] >= (uint64_t(1)));        // Size must be min 1
  assert(gmem_prob_shape[2] <= (uint64_t(1) << 32));  // Size must be max 2^32
  assert(gmem_prob_shape[3] >= (uint64_t(1)));        // Size must be min 1
  assert(gmem_prob_shape[3] <= (uint64_t(1) << 32));  // Size must be max 2^32
  assert(gmem_prob_shape[4] >= (uint64_t(1)));        // Size must be min 1
  assert(gmem_prob_shape[4] <= (uint64_t(1) << 32));  // Size must be max 2^32

  assert((gmem_prob_stride[0]) == sizeof(T));  // First stride is implicitly 1
  assert((gmem_prob_stride[1]) <
         (uint64_t(1) << 40));  // Stride must be max 2^40
  assert((gmem_prob_stride[1] & 0b1111) ==
         0);  // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[2]) <
         (uint64_t(1) << 40));  // Stride must be max 2^40
  assert((gmem_prob_stride[2] & 0b1111) ==
         0);  // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[3]) <
         (uint64_t(1) << 40));  // Stride must be max 2^40
  assert((gmem_prob_stride[3] & 0b1111) ==
         0);  // Stride must be multiple of 16B (128b)
  assert((gmem_prob_stride[4]) <
         (uint64_t(1) << 40));  // Stride must be max 2^40
  assert((gmem_prob_stride[4] & 0b1111) ==
         0);  // Stride must be multiple of 16B (128b)

  //
  // TMA smem desc info
  //

  // TMA smem box size
  cute::array<cuuint32_t, 5> smem_box_shape = {1, 1, 1, 1, 1};
  for_each(make_seq<rank(tma_layout_cta)>{}, [&](auto i) {
    uint32_t shape_i = shape<i>(tma_layout_cta);
    constexpr int tma_i = i < 5 ? i : 4;
    if (tma_multimode && tma_i == 4) {
      // We're "reusing" this TMA mode and using it as a "multimode"
      smem_box_shape[tma_i] = 1;
    } else {
      smem_box_shape[tma_i] = shape_i;
    }
  });

  // TMA smem mode strides
  [[maybe_unused]] cute::array<cuuint32_t, 5> smem_box_stride = {1, 1, 1, 1, 1};

  assert(smem_box_shape[0] >= (uint64_t(1)));       // Size must be min 1
  assert(smem_box_shape[0] <= (uint64_t(1) << 8));  // Size must be max 2^8
  assert(smem_box_shape[0] >= (uint64_t(1)));       // Size must be min 1
  assert(smem_box_shape[0] <= (uint64_t(1) << 8));  // Size must be max 2^8
  assert(smem_box_shape[0] >= (uint64_t(1)));       // Size must be min 1
  assert(smem_box_shape[0] <= (uint64_t(1) << 8));  // Size must be max 2^8
  assert(smem_box_shape[0] >= (uint64_t(1)));       // Size must be min 1
  assert(smem_box_shape[0] <= (uint64_t(1) << 8));  // Size must be max 2^8

  assert(smem_box_stride[0] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[0] <= (uint32_t(8)));  // Stride must be max 2^3
  assert(smem_box_stride[1] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[1] <= (uint32_t(8)));  // Stride must be max 2^3
  assert(smem_box_stride[2] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[2] <= (uint32_t(8)));  // Stride must be max 2^3
  assert(smem_box_stride[3] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[3] <= (uint32_t(8)));  // Stride must be max 2^3
  assert(smem_box_stride[4] >= (uint32_t(1)));  // Stride must be min 1
  assert(smem_box_stride[4] <= (uint32_t(8)));  // Stride must be max 2^3

  //
  // Construct the descriptor
  //

  TmaDescriptor tma_desc = {0};

#if (__CUDACC_VER_MAJOR__ >= 12)

  //
  // TMA general info
  //

  cuuint32_t tma_dim = TmaRANK;
  CUtensorMapDataType tma_format = TMA::to_CUtensorMapDataType<T>();
  CUtensorMapInterleave tma_interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapFloatOOBfill tma_oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // TMA smem swizzle type
  CUtensorMapSwizzle smem_swizzle =
      TMA::to_CUtensorMapSwizzle(get_tma_swizzle_bits(slayout));

  CUresult result = cuTensorMapEncodeTiled(
      &tma_desc,
      tma_format,
      tma_dim,
      gmem_address,
      gmem_prob_shape.data(),
      gmem_prob_stride.data() + 1,  // gmem_prob_stride[0] implicitly 1
      smem_box_shape.data(),
      smem_box_stride.data(),
      tma_interleave,
      smem_swizzle,
      tma_l2Promotion,
      tma_oobFill);

  if (result != CUDA_SUCCESS) {
    std::cerr << "TMA Desc Addr:   " << &tma_desc << "\nformat         "
              << tma_format << "\ndim            " << tma_dim
              << "\ngmem_address   " << gmem_address << "\nglobalDim      "
              << gmem_prob_shape << "\nglobalStrides  " << gmem_prob_stride
              << "\nboxDim         " << smem_box_shape << "\nelementStrides "
              << smem_box_stride << "\ninterleave     " << tma_interleave
              << "\nswizzle        " << smem_swizzle << "\nl2Promotion    "
              << tma_l2Promotion << "\noobFill        " << tma_oobFill
              << std::endl;
    std::cerr << "Error: Failed to intialize the TMA descriptor " << result
              << std::endl;
    assert(false);
  }
#endif  // (__CUDACC_VER_MAJOR__ >= 12)

  //
  // Construct the Copy_Traits
  //

  // Finally, get the inverse permutation of the E<i> bases for the mocked gmem
  // stride
  auto gmem_stride_bases_flat =
      transform(make_seq<rank(tma_layout_cta)>{}, [&](auto i) {
        auto k = find(stride(tma_layout_cta), E<i>{});
        // NOTE: gcc 7.3.5 WAR -- avoid if constexpr
        int32_t tma_coord_stride =
            int32_t(stride<i>(flat_glayout) * sizeof(T) /
                    (gmem_prob_stride[4] != 0 ? gmem_prob_stride[4] : 16));
        return conditional_return(
            tma_multimode && (k >= Int<4>{}),
            E<4>{} * tma_coord_stride,  // The 4th TMA mode is the multimode,
                                        // use int32_t coord stride
            E<k>{});
      });

  // Give that the profile of gtensor and fold it
  auto gmem_stride_bases =
      stride(composition(make_layout(repeat_like(shape(flat_glayout), Int<2>{}),
                                     gmem_stride_bases_flat),
                         make_layout(repeat_like(shape(gtensor), Int<2>{}))));

  constexpr int num_bits = size(sidx_to_gmode_cta_trunc) * sizeof(T) * 8;
  using Traits =
      Copy_Traits<CopyOp, Int<num_bits>, decltype(gmem_stride_bases)>;

#if 0
  print("num_bits      :  "); print(num_bits); print("\n");
  print("g_stride_bases:  "); print(gmem_stride_bases); print("\n");
#endif

  //
  // Construct the TiledCopy
  //

  // The ThrVal layout for 1 TMA instruction within cta_tile
  auto layout_tv_1 = composition(
      inv_smem_layout,
      make_layout(make_shape(cluster_size, size(sidx_to_gmode_cta_trunc)),
                  GenRowMajor{}));
  // The ThrVal layout for N TMA instructions within cta_tile
  auto layout_tv = tile_to_shape(
      layout_tv_1, make_shape(cluster_size, size(cta_tile) / cluster_size));

#if 0
  print("layout_tv     :  "); print(layout_tv); print("\n");
#endif

  return TiledCopy<Copy_Atom<Traits, T>,
                   decltype(layout_tv),
                   decltype(cta_tile)>{tma_desc, gmem_stride_bases};
}

// Explicit defaulting
template <class CopyOp, class GEngine, class GLayout, class SLayout>
CUTE_HOST auto make_tma_copy(CopyOp const& copy_op,
                             Tensor<GEngine, GLayout> const& gtensor,
                             SLayout const& slayout) {
  return make_tma_copy(
      copy_op, gtensor, slayout, product_each(shape(slayout)), Int<1>{});
}

template <class CopyOp,
          class GEngine,
          class GLayout,
          class SLayout,
          class Cluster_Size>
CUTE_HOST auto make_tma_copy(CopyOp const& copy_op,
                             Tensor<GEngine, GLayout> const& gtensor,
                             SLayout const& slayout,
                             Cluster_Size const& cluster_size) {
  return make_tma_copy(
      copy_op, gtensor, slayout, product_each(shape(slayout)), cluster_size);
}

}  // end namespace cute
