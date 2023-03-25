// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//  Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
//
//  This source code is licensed under the BSD license found in the
//  LICENSE file in the root directory of this source tree.

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

#pragma once

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
  using DefaultMma =
      cutlass::gemm::threadblock::DefaultMma<ElementA,
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
template <typename ElementA_,
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

  using DefaultMma_ = cutlass::gemm::threadblock::DefaultMma<ElementA_,
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

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
