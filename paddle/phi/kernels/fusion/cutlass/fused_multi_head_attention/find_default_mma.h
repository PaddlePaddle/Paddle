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
 * 3. Neither the name of the copyright holdvr nor the names of its
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
    \brief Cutlass provides helper template functions to figure out the right
   datastructures to instanciate to run a GEMM with various parameters (see
   `cutlass/gemm/threadblock/default_mma.h`). However, due to template
   instantiation priority rules, it will only create an MmaMultiStage with
   kStages=3 (otherwise creates an MmePipelined - which is not compatible with
   FastF32). kStages=3 uses too much shared memory and we want to use kStages=2,
   so we just copy-pasted some code from `default_mma.h` and
   `default_mma_core.h` files and wrapped this template to allow our usecase.

    This is really only for the FastF32 case - aka using TensorCores with fp32.
*/

#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    typename Enable_ = void>
struct FindDefaultMma {
  static constexpr bool AccumulatorsInRowMajor = false;
  static constexpr SharedMemoryClearOption SharedMemoryClear =
      SharedMemoryClearOption::kNone;
  using DefaultMma = cutlass::gemm::threadblock::DefaultMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      Stages,
      Operator,
      AccumulatorsInRowMajor,
      SharedMemoryClear>;
};

/// Specialization for sm80 / FastF32 / multistage with kStages=2
template <
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    int kStages,
    typename Operator>
struct FindDefaultMma<
    ElementA_,
    LayoutA_,
    kAlignmentA,
    ElementB_,
    LayoutB_,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm80,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    kStages,
    Operator,
    typename cutlass::platform::enable_if<(kAlignmentA > 1)>::type> {
  using LayoutC = layout::RowMajor;
  using OperatorClass = arch::OpClassTensorOp;
  using ArchTag = arch::Sm80;

  using DefaultMma_ = cutlass::gemm::threadblock::DefaultMma<
      ElementA_,
      LayoutA_,
      kAlignmentA,
      ElementB_,
      LayoutB_,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      3,
      Operator>;
  struct DefaultMma : DefaultMma_ {
    using MmaCore_ = typename DefaultMma_::MmaCore;
    // Define the threadblock-scoped multistage matrix multiply
    using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
        typename MmaCore_::Shape,
        typename DefaultMma_::IteratorA,
        typename MmaCore_::SmemIteratorA,
        MmaCore_::kCacheOpA,
        typename DefaultMma_::IteratorB,
        typename MmaCore_::SmemIteratorB,
        MmaCore_::kCacheOpB,
        ElementAccumulator,
        LayoutC,
        typename MmaCore_::MmaPolicy,
        kStages>;
  };
};

} // namespace threadblock
} // namespace gemm
} // namespace cutlass
