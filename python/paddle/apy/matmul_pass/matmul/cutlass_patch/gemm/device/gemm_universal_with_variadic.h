// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

/*! \file
    \brief
*/

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/gemm/device/default_gemm_configuration.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/default_gemm_universal.h"
#include "cutlass_patch/gemm/kernel/default_gemm_with_variadic.h"

#include "cutlass/layout/permute.h"

namespace cutlass {
namespace gemm {
namespace device {

/*!
  GemmUniversal with variadic epilogues.
*/
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_,
        ArchTag_,
        ElementA_,
        ElementB_,
        ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ =
        typename DefaultGemmConfiguration<OperatorClass_,
                                          ArchTag_,
                                          ElementA_,
                                          ElementB_,
                                          ElementC_,
                                          ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_,
        ArchTag_,
        ElementA_,
        ElementB_,
        ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_,
        ArchTag_,
        ElementA_,
        ElementB_,
        ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages = DefaultGemmConfiguration<OperatorClass_,
                                          ArchTag_,
                                          ElementA_,
                                          ElementB_,
                                          ElementC_,
                                          ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = DefaultGemmConfiguration<OperatorClass_,
                                              ArchTag_,
                                              ElementA_,
                                              ElementB_,
                                              ElementC_,
                                              ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = DefaultGemmConfiguration<OperatorClass_,
                                              ArchTag_,
                                              ElementA_,
                                              ElementB_,
                                              ElementC_,
                                              ElementAccumulator_>::kAlignmentB,
    /// Operation performed by GEMM
    typename Operator_ =
        typename DefaultGemmConfiguration<OperatorClass_,
                                          ArchTag_,
                                          ElementA_,
                                          ElementB_,
                                          ElementC_,
                                          ElementAccumulator_>::Operator,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA = ComplexTransform::kNone,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB = ComplexTransform::kNone>
class GemmUniversalWithVariadic
    : public GemmUniversalBase<
          typename kernel::DefaultGemmWithVariadic<ElementA_,
                                                   LayoutA_,
                                                   TransformA,
                                                   AlignmentA,
                                                   ElementB_,
                                                   LayoutB_,
                                                   TransformB,
                                                   AlignmentB,
                                                   ElementC_,
                                                   LayoutC_,
                                                   ElementAccumulator_,
                                                   OperatorClass_,
                                                   ArchTag_,
                                                   ThreadblockShape_,
                                                   WarpShape_,
                                                   InstructionShape_,
                                                   EpilogueOutputOp_,
                                                   ThreadblockSwizzle_,
                                                   Stages,
                                                   Operator_>::GemmKernel> {
 public:
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;

  using Base = GemmUniversalBase<
      typename kernel::DefaultGemmWithVariadic<ElementA_,
                                               LayoutA_,
                                               TransformA,
                                               AlignmentA,
                                               ElementB_,
                                               LayoutB_,
                                               TransformB,
                                               AlignmentB,
                                               ElementC_,
                                               LayoutC_,
                                               ElementAccumulator_,
                                               OperatorClass_,
                                               ArchTag_,
                                               ThreadblockShape_,
                                               WarpShape_,
                                               InstructionShape_,
                                               EpilogueOutputOp_,
                                               ThreadblockSwizzle_,
                                               Stages,
                                               Operator_>::GemmKernel>;

  using Arguments = typename Base::Arguments;
  using GemmKernel = typename Base::GemmKernel;
};

/// Partial specialization for column-major output exchanges problem size and
/// operand.
template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for.  This is the minimum SM that
    /// supports the intended feature. The device kernel can be built
    /// targeting any SM larger than this number.
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// Operation performed by GEMM
    typename Operator_,
    /// Complex elementwise transformation on A operand
    ComplexTransform TransformA,
    /// Complex elementwise transformation on B operand
    ComplexTransform TransformB>
class GemmUniversalWithVariadic<ElementA_,
                                LayoutA_,
                                ElementB_,
                                LayoutB_,
                                ElementC_,
                                layout::ColumnMajor,  // partially specialized
                                                      // on LayoutC
                                ElementAccumulator_,
                                OperatorClass_,
                                ArchTag_,
                                ThreadblockShape_,
                                WarpShape_,
                                InstructionShape_,
                                EpilogueOutputOp_,
                                ThreadblockSwizzle_,
                                Stages,
                                AlignmentA,
                                AlignmentB,
                                Operator_,
                                TransformA,
                                TransformB> {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static ComplexTransform const kTransformA = TransformA;
  static ComplexTransform const kTransformB = TransformB;

  using UnderlyingOperator = typename GemmUniversalWithVariadic<
      ElementB,
      typename layout::LayoutTranspose<LayoutB>::type,
      ElementA,
      typename layout::LayoutTranspose<LayoutA>::type,
      ElementC,
      layout::RowMajor,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      Stages,
      kAlignmentB,
      kAlignmentA,
      Operator,
      kTransformB,
      kTransformA>::Base;

  using GemmKernel = typename UnderlyingOperator::GemmKernel;
  static int const kAlignmentC = EpilogueOutputOp::kCount;

  /// Argument structure
  using Arguments = typename UnderlyingOperator::Arguments;

 private:
  UnderlyingOperator underlying_operator_;

 public:
  /// Constructs the GEMM.
  GemmUniversalWithVariadic() {}

  /// Helper to construct a transposed equivalent for the underlying GEMM
  /// operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    return args.transposed_problem();
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    return UnderlyingOperator::get_workspace_size(
        to_underlying_arguments(args));
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {
    return UnderlyingOperator::get_grid_shape(to_underlying_arguments(args));
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    return UnderlyingOperator::maximum_active_blocks(smem_capacity);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args,
                    void *workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    return underlying_operator_.initialize(
        to_underlying_arguments(args), workspace, stream);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    return underlying_operator_.update(to_underlying_arguments(args),
                                       workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {
    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) { return run(stream); }

  /// Runs the kernel using initialized state.
  Status operator()(Arguments const &args,
                    void *workspace = nullptr,
                    cudaStream_t stream = nullptr) {
    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

}  // namespace device
}  // namespace gemm
}  // namespace cutlass
