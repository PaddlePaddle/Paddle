/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

/*! \file
    \brief
      Default kernel-level GEMM definitions combine threadblock-scoped matrix
   multiply-add with the appropriate threadblock-scoped epilogue.

      Note, CUTLASS epilogues universally target row-major outputs. Column-major
   outputs are accommodated by exchanging A and B operands and assuming
   transposed layouts. Partial specializations here choose
   'device::GemmTransposed' to implement this functionality.

*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/complex.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"

#include "./gemm_kernel_utils.h"
#include "gemm/attention_scaling_coefs_updater.h"
#include "gemm/find_default_mma.h"
#include "gemm/fmha_grouped.h"
#include "gemm/mma_from_smem.h"
#include "transform/tile_smem_loader.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    // The datatype of Q/K/V
    typename scalar_t_,
    // Architecture we are targeting (eg `cutlass::arch::Sm80`)
    typename ArchTag_,
    // If Q/K/V are correctly aligned in memory and we can run a fast kernel
    bool isAligned_,
    bool maskIsAligned_,
    int QueriesPerBlock_,
    int KeysPerBlock_,
    bool SingleValueIteration_,
    GroupScheduleMode GroupScheduleMode_,
    bool AddMask>
struct DefaultFMHAGrouped {
  using scalar_t = scalar_t_;
  using accum_t = float;
  using output_t = scalar_t;

  // Accumulator between 2 iterations
  // Using `accum_t` improves perf on f16 at the cost of
  // numerical errors
  using output_accum_t = accum_t;

  using ArchTag = ArchTag_;
  static bool const kIsAligned = isAligned_;
  static bool const kAddMask = AddMask;
  static bool const kSingleValueIteration = SingleValueIteration_;
  static int const kKeysPerBlock = KeysPerBlock_;
  static bool const kMaskIsAligned = maskIsAligned_;
  static int const kWarpSize = 32;
  static int const kNumWarpsPerBlock =
      QueriesPerBlock_ * KeysPerBlock_ / (kWarpSize * kWarpSize);

  struct MM0 {
    /*
      In this first matmul, we compute a block of `Q @ K.T`.
      While the calculation result is still hot in registers, we update
      `mi`, `m_prime`, `s_prime` in shared-memory, and then store this value
      into a shared-memory ("AccumulatorSharedStorage") that is used later as
      operand A for the second matmul (see MM1)
    */

    using GemmType = gemm_kernel_utils::DefaultGemmType<ArchTag, scalar_t>;
    using OpClass = typename GemmType::OpClass;

    using ElementA = scalar_t;
    using ElementB = scalar_t;
    using ElementC = scalar_t;
    using ElementAccumulator = accum_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            ElementA,
            ElementB,
            ElementC,
            ElementAccumulator>;

    static int const kAlignmentA =
        kIsAligned ? DefaultConfig::kAlignmentA : GemmType::kMinimumAlignment;
    static int const kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;

    using ThreadblockShape = cutlass::gemm::
        GemmShape<QueriesPerBlock_, KeysPerBlock_, GemmType::ThreadK>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, GemmType::WarpK>;
    using InstructionShape = typename GemmType::InstructionShape;

    static int const kStages = DefaultConfig::kStages;
    using Operator = typename GemmType::Operator;

    using DefaultMma = typename cutlass::gemm::threadblock::FindDefaultMma<
        ElementA,
        LayoutA,
        kAlignmentA,
        ElementB,
        LayoutB,
        kAlignmentB,
        ElementAccumulator,
        LayoutC,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        kStages,
        Operator>::DefaultMma;

    using MmaCore = typename DefaultMma::MmaCore;
    using IteratorA = typename DefaultMma::IteratorA;
    using IteratorB = typename DefaultMma::IteratorB;
    using Mma = typename DefaultMma::ThreadblockMma;
    using ScalingCoefsUpdater = typename DefaultAttentionScalingCoefsUpdater<
        typename Mma::Operator::IteratorC,
        ElementAccumulator,
        kWarpSize>::Updater;
    static_assert(MmaCore::WarpCount::kCount == kNumWarpsPerBlock, "");

    // used for efficient load of mask_ tile Bij from global to shared memory
    using MaskLoader = TileSmemLoader<
        scalar_t,
        cutlass::MatrixShape<QueriesPerBlock_, KeysPerBlock_>,
        MmaCore::kThreads,
        // input restriction: kv_len has to be a multiple of this value
        kMaskIsAligned ? 128 / cutlass::sizeof_bits<scalar_t>::value : 1>;

    // Epilogue to store to shared-memory in a format that we can use later for
    // the second matmul
    using B2bGemm = typename cutlass::gemm::threadblock::B2bGemm<
        typename Mma::Operator::IteratorC,
        typename Mma::Operator,
        scalar_t,
        WarpShape,
        ThreadblockShape>;
    using AccumulatorSharedStorage = typename B2bGemm::AccumulatorSharedStorage;
  };

  struct MM1 {
    /*
      Second matmul: perform `attn @ V` where `attn` is the attention (not
      normalized) and stored in shared memory
    */

    using GemmType = typename MM0::GemmType;
    using OpClass = typename GemmType::OpClass;

    using ElementA = scalar_t;
    using ElementB = scalar_t;
    using ElementC = output_accum_t;
    using ElementAccumulator = accum_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    using DefaultConfig =
        typename cutlass::gemm::device::DefaultGemmConfiguration<
            OpClass,
            ArchTag,
            ElementA,
            ElementB,
            ElementC,
            ElementAccumulator>;

    static int const kAlignmentA = DefaultConfig::kAlignmentA;
    static int const kAlignmentB =
        kIsAligned ? DefaultConfig::kAlignmentB : GemmType::kMinimumAlignment;

    using ThreadblockShape = typename MM0::ThreadblockShape;
    using WarpShape = typename MM0::WarpShape;
    using InstructionShape = typename MM0::InstructionShape;

    using EpilogueOutputOp = typename DefaultConfig::EpilogueOutputOp;

    static int const kStages = DefaultConfig::kStages;
    using Operator = typename GemmType::Operator;

    using ThreadblockSwizzle = void;  // Swizzling is unused
    static bool const kSplitKSerial = false;

    using DefaultGemm = cutlass::gemm::kernel::DefaultGemm<ElementA,
                                                           LayoutA,
                                                           kAlignmentA,
                                                           ElementB,
                                                           LayoutB,
                                                           kAlignmentB,
                                                           ElementC,
                                                           LayoutC,
                                                           ElementAccumulator,
                                                           OpClass,
                                                           ArchTag,
                                                           ThreadblockShape,
                                                           WarpShape,
                                                           InstructionShape,
                                                           EpilogueOutputOp,
                                                           ThreadblockSwizzle,
                                                           kStages,
                                                           kSplitKSerial,
                                                           Operator>;

    using DefaultMmaFromSmem =
        typename cutlass::gemm::threadblock::DefaultMmaFromSharedMemory<
            typename DefaultGemm::Mma,
            typename MM0::AccumulatorSharedStorage,
            false>;

    using Mma = typename DefaultMmaFromSmem::Mma;
    using IteratorB = typename Mma::IteratorB;
    using WarpCount = typename Mma::WarpCount;
    static_assert(WarpCount::kCount == kNumWarpsPerBlock, "");

    using DefaultEpilogue = typename DefaultGemm::Epilogue;
    using OutputTileIterator =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_t>;
    using OutputTileIteratorAccum =
        typename cutlass::epilogue::threadblock::PredicatedTileIterator<
            typename DefaultEpilogue::OutputTileIterator::ThreadMap,
            output_accum_t>;

    struct SharedStorageMM1 {
      typename Mma::SharedStorage mm;
    };
  };

  /// Define the kernel in terms of the default kernel
  using FMHAKernel = kernel::FMHAGrouped<MM0,
                                         MM1,
                                         scalar_t,
                                         accum_t,
                                         output_t,
                                         output_accum_t,
                                         SingleValueIteration_,
                                         GroupScheduleMode_,
                                         AddMask,
                                         maskIsAligned_>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
