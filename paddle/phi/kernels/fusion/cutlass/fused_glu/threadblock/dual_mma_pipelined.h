/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/mma_base.h"
#include "dual_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Transformation applied to A operand
    typename TransformA_ = NumericArrayConverter<
        typename SmemIteratorA_::Element,
        typename IteratorA_::Element,
        IteratorA_::Fragment::kElements>,
    ///
    /// Transformation applied to B operand
    typename TransformB_ = NumericArrayConverter<
        typename SmemIteratorB_::Element,
        typename IteratorB_::Element,
        IteratorB_::Fragment::kElements>,
    /// Used for partial specialization
    typename Enable = bool>
class DualMmaPipelined : 
  public DualMmaBase<Shape_, Policy_, 2> {
public:
  ///< Base class
  using Base = DualMmaBase<Shape_, Policy_, 2>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  
  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = typename Policy::Operator::ArchTag;
  
  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
  static_assert(
      (Base::kStages == 2),
      "MmaPipelined requires kStages set to value 2");

  static bool const kSmemContainsEntireMat = false;

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B0_;
  SmemIteratorB smem_iterator_B1_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  DualMmaPipelined(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B0_(shared_storage.operand_B0_ref(), thread_idx),
      smem_iterator_B1_(shared_storage.operand_B1_ref(), thread_idx)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B0_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    this->warp_tile_iterator_B1_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum0,
      FragmentC &accum1,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      ///< iterator over B operand in global memory
      IteratorB iterator_B0,
      IteratorB iterator_B1,
      ///< initial value of accumulator
      FragmentC const &src_accum0,
      TransformA transform_A =
          TransformA(), ///< transformation applied to A fragment
      TransformB transform_B =
          TransformB()) { ///< transformation applied to B fragment

        // TODO(zhengzekang): transformer_C？
    
    // Perform accumulation in the 'd' output operand
    accum0 = src_accum0;
    accum1 = src_accum1;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B0;
    FragmentB tb_frag_B1;

    tb_frag_A.clear();
    tb_frag_B0.clear();
    tb_frag_B1.clear();

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B0.load(tb_frag_B0);
    iterator_B1.load(tb_frag_B1);

    ++iterator_A;
    ++iterator_B0;
    ++iterator_B1;

    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    this->smem_iterator_B0_.store(transform_B(tb_frag_B0));
    this->smem_iterator_B1_.store(transform_B(tb_frag_B1));

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B0_;
    ++this->smem_iterator_B1_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B0[2];
    WarpFragmentB warp_frag_B1[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B0_.set_kgroup_index(0);
    this->warp_tile_iterator_B1_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B0_.load(warp_frag_B0[0]);
    this->warp_tile_iterator_B1_.load(warp_frag_B1[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B0_;
    ++this->warp_tile_iterator_B1_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B0.clear_mask(gemm_k_iterations <= 1);
    iterator_B1.clear_mask(gemm_k_iterations <= 1);


    // Issue loads during the first warp-level matrix multiply-add *AFTER*
    // issuing shared memory loads (which have the tighest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {
        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {
          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B0_.store(transform_B(tb_frag_B0));
          this->smem_iterator_B1_.store(transform_B(tb_frag_B1));


          __syncthreads();

          ++this->smem_iterator_A_;
          ++this->smem_iterator_B0_;
          ++this->smem_iterator_B1_;

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B0_.add_tile_offset({-Base::kStages, 0});
            this->smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
          } else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0,
                 -Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations});
            this->warp_tile_iterator_B0_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            this->warp_tile_iterator_B1_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B0_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B1_.set_kgroup_index(
            (warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B0_.load(warp_frag_B0[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B1_.load(warp_frag_B1[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B0_;
        ++this->warp_tile_iterator_B1_;

        if (warp_mma_k == 0) {
          iterator_A.load(tb_frag_A);
          iterator_B0.load(tb_frag_B0);
          iterator_B1.load(tb_frag_B1);

          ++iterator_A;
          ++iterator_B0;
          ++iterator_B1;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B0.clear_mask(gemm_k_iterations <= 2);
          iterator_B1.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(
            accum0,
            warp_frag_A[warp_mma_k % 2],
            warp_frag_B0[warp_mma_k % 2],
            accum0);
        
        warp_mma(
            accum1,
            warp_frag_A[warp_mma_k % 2],
            warp_frag_B1[warp_mma_k % 2],
            accum1);
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
