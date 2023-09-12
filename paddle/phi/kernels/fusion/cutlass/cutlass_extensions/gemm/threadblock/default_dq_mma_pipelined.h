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

#pragma once

#include "cutlass/gemm/threadblock/default_mma.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/arch/mma.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/dq_mma_pipelined.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/tile_interleaved_layout.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_dq_mma.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    /// Operation performed by GEMM
    typename Operator>
struct DqMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementScale,
    LayoutScale,
    kAlignmentScale,
    ElementAccumulator,
    layout::RowMajor,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    2,
    Operator,
    SharedMemoryClearOption::kNone,
    typename platform::enable_if<(ArchTag::kMinComputeCapability < 80)>::type> {
  static_assert(platform::is_same<ElementA, half_t>::value ||
                    platform::is_same<ElementA, bfloat16_t>::value,
                "Element A must be fp16 or bf16");

  static_assert(platform::is_same<ElementB, uint8_t>::value ||
                    platform::is_same<ElementB, uint4b_t>::value,
                "Element B must be int8 or uint4");

  static constexpr bool DqAfterLDG =
      platform::is_same<arch::OpMultiplyAdd, Operator>::value;
  static constexpr bool arch_has_bf16_mma =
      ArchTag::kMinComputeCapability >= 80;
  using MmaCoreElementA =
      typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;
  using MmaCoreElementB = typename platform::
      conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;

  // Define the MmaCore components
  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          MmaCoreElementA,
                                                          LayoutA,
                                                          MmaCoreElementB,
                                                          LayoutB,
                                                          ElementAccumulator,
                                                          layout::RowMajor,
                                                          OperatorClass,
                                                          2,
                                                          Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
      ElementA,
      LayoutA,
      1,
      typename MmaCore::IteratorThreadMapA,
      kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
      ElementB,
      LayoutB,
      0,
      typename MmaCore::IteratorThreadMapB,
      kAlignmentB>;

  // ThreadMap for scale iterator
  static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
  using IteratorScaleThreadMap = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
      MmaCore::Shape::kN / kAlignmentScale,
      kAlignmentScale>;

  // Define iterators over tiles from the scale operand
  using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<1, MmaCore::Shape::kN>,
      ElementScale,
      LayoutScale,
      0,
      IteratorScaleThreadMap,
      kAlignmentScale>;

  using SmemScaleType = typename platform::
      conditional<arch_has_bf16_mma, ElementScale, half_t>::type;
  using SmemIteratorScale =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<1, MmaCore::Shape::kN>,
          SmemScaleType,
          LayoutScale,
          0,
          IteratorScaleThreadMap,
          kAlignmentScale>;

  using Converters =
      SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<
      typename MmaCore::Shape,
      IteratorA,
      typename MmaCore::SmemIteratorA,
      IteratorB,
      typename MmaCore::SmemIteratorB,
      IteratorScale,
      SmemIteratorScale,
      ElementAccumulator,
      layout::RowMajor,
      typename MmaCore::MmaPolicy,
      typename Converters::TransformAfterLDG,
      typename Converters::TransformAfterLDS>;
};

// Specialization to handle column major interleave B
template <
    /// Type for element A
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Type for element B
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for the input scale
    typename ElementScale,
    /// Layout for the scale operand
    typename LayoutScale,
    /// Access granularity of Scales in unit of elements
    int kAlignmentScale,
    /// Element type for internal accumulation
    typename ElementAccumulator,
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
    /// Operation performed by GEMM
    typename Operator,
    ///
    int RowsPerTile,
    ///
    int ColumnsInterleaved>
struct DqMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    layout::ColumnMajorTileInterleave<RowsPerTile, ColumnsInterleaved>,
    kAlignmentB,
    ElementScale,
    LayoutScale,
    kAlignmentScale,
    ElementAccumulator,
    layout::RowMajor,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    2,
    Operator,
    SharedMemoryClearOption::kNone,
    typename platform::enable_if<(ArchTag::kMinComputeCapability < 80)>::type> {
  static_assert(platform::is_same<ElementA, half_t>::value ||
                    platform::is_same<ElementA, bfloat16_t>::value,
                "Element A must be fp16 or bf16");

  static_assert(platform::is_same<ElementB, uint8_t>::value ||
                    platform::is_same<ElementB, uint4b_t>::value,
                "Element B must be int8 or uint4");

  static constexpr bool DqAfterLDG =
      platform::is_same<arch::OpMultiplyAdd, Operator>::value;
  static constexpr bool arch_has_bf16_mma =
      ArchTag::kMinComputeCapability >= 80;
  using MmaCoreElementA =
      typename platform::conditional<arch_has_bf16_mma, ElementA, half_t>::type;
  using MmaCoreElementB = typename platform::
      conditional<DqAfterLDG, MmaCoreElementA, ElementB>::type;

  // Define the MmaCore components
  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          MmaCoreElementA,
                                                          LayoutA,
                                                          MmaCoreElementB,
                                                          layout::ColumnMajor,
                                                          ElementAccumulator,
                                                          layout::RowMajor,
                                                          OperatorClass,
                                                          2,
                                                          Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
      ElementA,
      LayoutA,
      1,
      typename MmaCore::IteratorThreadMapA,
      kAlignmentA>;

 private:
  static_assert(!(MmaCore::Shape::kN % ColumnsInterleaved), "");
  static_assert(RowsPerTile == MmaCore::Shape::kK, "");

  using OriginalThreadMap = typename MmaCore::IteratorThreadMapB;
  using OriginalWarpArrangement =
      typename OriginalThreadMap::Detail::WarpThreadArrangement;
  static_assert(!(OriginalWarpArrangement::kStrided % ColumnsInterleaved), "");

  using GmemIteratorShape =
      MatrixShape<MmaCore::Shape::kK * ColumnsInterleaved,
                  MmaCore::Shape::kN / ColumnsInterleaved>;
  using GmemThreadMapB = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<GmemIteratorShape::kRow,
                               GmemIteratorShape::kColumn>,
      OriginalThreadMap::kThreads,
      layout::PitchLinearShape<
          OriginalWarpArrangement::kContiguous * ColumnsInterleaved,
          OriginalWarpArrangement::kStrided / ColumnsInterleaved>,
      MmaCore::kAccessSizeInBits / sizeof_bits<ElementB>::value>;

 public:
  // Define iterators over tiles from the B operand
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      GmemIteratorShape,
      ElementB,
      layout::ColumnMajor,
      0,
      GmemThreadMapB,
      kAlignmentB>;

  // ThreadMap for scale iterator
  static_assert((MmaCore::Shape::kN % kAlignmentScale) == 0, "");
  using IteratorScaleThreadMap = transform::PitchLinearStripminedThreadMap<
      layout::PitchLinearShape<MmaCore::Shape::kN, 1>,
      MmaCore::Shape::kN / kAlignmentScale,
      kAlignmentScale>;

  // Define iterators over tiles from the scale operand
  using IteratorScale = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<1, MmaCore::Shape::kN>,
      ElementScale,
      LayoutScale,
      0,
      IteratorScaleThreadMap,
      kAlignmentScale>;

  using SmemScaleType = typename platform::
      conditional<arch_has_bf16_mma, ElementScale, half_t>::type;
  using SmemIteratorScale =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<1, MmaCore::Shape::kN>,
          SmemScaleType,
          LayoutScale,
          0,
          IteratorScaleThreadMap,
          kAlignmentScale>;

  using Converters =
      SetConverters<IteratorB, typename MmaCore::MmaPolicy::Operator, Operator>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::DqMmaPipelined<
      typename MmaCore::Shape,
      IteratorA,
      typename MmaCore::SmemIteratorA,
      IteratorB,
      typename MmaCore::SmemIteratorB,
      IteratorScale,
      SmemIteratorScale,
      ElementAccumulator,
      layout::RowMajor,
      typename MmaCore::MmaPolicy,
      typename Converters::TransformAfterLDG,
      typename Converters::TransformAfterLDS>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
