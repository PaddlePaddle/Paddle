/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
/*! \file
    \brief Templates implementing warp-level matrix multiply-accumulate
   operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/platform/platform.h"

#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/arch/mma_sm75.h"
#include "cutlass/arch/mma_sm80.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"

#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator_sm80.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Data type of A elements
    typename ElementA_,
    /// Layout of A matrix (concept: MatrixLayout)
    typename LayoutA_,
    /// Data type of B elements
    typename ElementB_,
    /// Layout of B matrix (concept: MatrixLayout)
    typename LayoutB_,
    /// Element type of C matrix
    typename ElementC_,
    /// Layout of C matrix (concept: MatrixLayout)
    typename LayoutC_,
    /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
    typename Policy_,
    /// Instruction shape to override shared memory iterators with
    typename SharedMemoryInstructionShape_,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Used for partial specialization
    typename Enable = bool>
class MmaTensorOpComputeBWithF16 {
 public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Indicates math operator
  using MathOperator = typename ArchMmaOperator::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;
  static_assert(
      (platform::is_same<typename ArchMmaOperator::ElementA, half_t>::value &&
       platform::is_same<typename ArchMmaOperator::ElementB, half_t>::value) ||
          (platform::is_same<typename ArchMmaOperator::ElementA,
                             bfloat16_t>::value &&
           platform::is_same<typename ArchMmaOperator::ElementB,
                             bfloat16_t>::value &&
           ArchTag::kMinComputeCapability >= 80),
      "MmaTensorOpCvtBToA only supports underlying HMMA");

  static_assert(platform::is_same<ElementA, half_t>::value ||
                    (platform::is_same<ElementA, bfloat16_t>::value &&
                     ArchTag::kMinComputeCapability >= 80),
                "MmaTensorOpCvtBToA only supports Fp16 A or Bf16 A on Ampere+");

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Instruction shape to override shared memory iterators with
  using SharedMemoryInstructionShape = SharedMemoryInstructionShape_;

  static_assert(SharedMemoryInstructionShape::kM == InstructionShape::kM,
                "M dimension of compute instruction must match load");
  static_assert(SharedMemoryInstructionShape::kN == InstructionShape::kN,
                "N dimension of compute instruction must match load");

  static constexpr int kExpansionFactor =
      SharedMemoryInstructionShape::kK / InstructionShape::kK;

  static_assert(!(Shape::kK % SharedMemoryInstructionShape::kK), "");

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

 public:
  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kM, Shape::kK>,
      Operand::kA,
      ElementA,
      LayoutA,
      MatrixShape<InstructionShape::kM, InstructionShape::kK>,
      Policy::OpDelta::kRow,
      kThreadCount,
      kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
      MatrixShape<Shape::kK, Shape::kN>,
      Operand::kB,
      ElementB,
      LayoutB,
      MatrixShape<SharedMemoryInstructionShape::kK, InstructionShape::kN>,
      Policy::OpDelta::kRow,
      kThreadCount,
      kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Iterates over the C operand in memory
  using IteratorC =
      MmaTensorOpAccumulatorTileIterator<MatrixShape<Shape::kM, Shape::kN>,
                                         ElementC,
                                         LayoutC,
                                         typename ArchMmaOperator::Shape,
                                         typename Policy::OpDelta>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  /// Number of mma operations performed
  using MmaIterations =
      MatrixShape<(Shape::kM + ArchMmaOperator::Shape::kM - 1) /
                      ArchMmaOperator::Shape::kM,
                  (Shape::kN + ArchMmaOperator::Shape::kN - 1) /
                      ArchMmaOperator::Shape::kN>;

 public:
  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOpComputeBWithF16() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(FragmentC& D,  // NOLINT
                  TransformedFragmentA const& A,
                  TransformedFragmentB const& B,
                  FragmentC const& C,
                  const int warp_tileB_k_offset) const {
    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    static_assert(
        TransformedFragmentB::kElements ==
            MmaOperandB::kElements * kExpansionFactor * MmaIterations::kColumn,
        "Each thread should have a pack of mma registers for each column "
        "iteration AND for the expanded K dim of B");

    D = C;

    MmaOperandA const* ptr_A = reinterpret_cast<MmaOperandA const*>(&A);
    MmaOperandB const* ptr_B = reinterpret_cast<MmaOperandB const*>(&B);
    MmaOperandC* ptr_D = reinterpret_cast<MmaOperandC*>(&D);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    // Serpentine visitation order maximizing reuse of Rb
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < MmaIterations::kColumn; ++n) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kRow; ++m) {
        int m_serpentine = ((n % 2) ? (MmaIterations::kRow - 1 - m) : m);

        int n_offsetB = warp_tileB_k_offset + kExpansionFactor * n;
        if (AccumulatorsInRowMajor) {  // matrix B is reordered
          mma(ptr_D[n + m_serpentine * MmaIterations::kColumn],
              ptr_A[m_serpentine],
              ptr_B[n_offsetB],
              ptr_D[n + m_serpentine * MmaIterations::kColumn]);
        } else {
          mma(ptr_D[m_serpentine + n * MmaIterations::kRow],
              ptr_A[m_serpentine],
              ptr_B[n_offsetB],
              ptr_D[m_serpentine + n * MmaIterations::kRow]);
        }
      }
    }
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    // Serpentine visitation order maximizing reuse of Ra
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < MmaIterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < MmaIterations::kColumn; ++n) {
        int n_serpentine = ((m % 2) ? (MmaIterations::kColumn - 1 - n) : n);

        int n_serpentine_offsetB =
            warp_tileB_k_offset + kExpansionFactor * n_serpentine;
        if (AccumulatorsInRowMajor) {  // matrix B is reordered
          mma(ptr_D[n_serpentine + m * MmaIterations::kColumn],
              ptr_A[m],
              ptr_B[n_serpentine_offsetB],
              ptr_D[n_serpentine + m * MmaIterations::kColumn]);
        } else {
          mma(ptr_D[m + n_serpentine * MmaIterations::kRow],
              ptr_A[m],
              ptr_B[n_serpentine_offsetB],
              ptr_D[m + n_serpentine * MmaIterations::kRow]);
        }
      }
    }
#else
    assert(0);
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
