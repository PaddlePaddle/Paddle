/***************************************************************************************************
 * Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
// 
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/blas3.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_triangular_matrix.h"
#include "cutlass/gemm/threadblock/mma_blas3_multistage.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Side Mode for the kernel
    SideMode kSideMode,
    /// Fill Mode for the triangular matrix
    FillMode kFillMode,
    /// Diag Type for the triangular matrix
    DiagType kDiagType,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false
    >
struct DefaultTrmm;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
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
    /// Side Mode for the kernel
    SideMode kSideMode,
    /// Fill Mode for the triangular matrix
    FillMode kFillMode,
    /// Diag Type for the triangular matrix
    DiagType kDiagType,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator
    >
struct DefaultTrmm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, 
                  kSideMode, kFillMode, kDiagType, 
                  ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, kSideMode, kFillMode, kDiagType, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, kSideMode, FillMode::kFull, DiagType::kInvalid, AccessTypeB>;
  
  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClearOption::kZfill>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output, right side mode (OperatorClass TensorOp)
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
    /// Fill Mode for the triangular matrix
    FillMode kFillMode,
    /// Diag Type for the triangular matrix
    DiagType kDiagType,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator
    >
struct DefaultTrmm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, 
                  SideMode::kRight, kFillMode, kDiagType, 
                  ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, SideMode::kRight, FillMode::kFull, DiagType::kInvalid, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, SideMode::kRight, kFillMode, kDiagType, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClearOption::kZfill>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output with unit diagonal (OperatorClass TensorOp)
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
    /// Side Mode for the kernel
    SideMode kSideMode,
    /// Fill Mode for the triangular matrix
    FillMode kFillMode,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator
    >
struct DefaultTrmm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, 
                  kSideMode, kFillMode, DiagType::kUnit, 
                  ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, kSideMode, kFillMode, DiagType::kUnit, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, kSideMode, FillMode::kFull, DiagType::kInvalid, AccessTypeB>;
  
  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaBlas3Multistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClearOption::kZfill>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output, right side mode, unit diagonal (OperatorClass TensorOp)
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
    /// Fill Mode for the triangular matrix
    FillMode kFillMode,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator
    >
struct DefaultTrmm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, 
                  SideMode::kRight, kFillMode, DiagType::kUnit, 
                  ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, Stages, Operator, false> {

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;

  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, SideMode::kRight, FillMode::kFull, DiagType::kInvalid, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;

  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIteratorTriangularMatrix<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, SideMode::kRight, kFillMode, DiagType::kUnit, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaBlas3Multistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClearOption::kZfill>;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass 

////////////////////////////////////////////////////////////////////////////////
