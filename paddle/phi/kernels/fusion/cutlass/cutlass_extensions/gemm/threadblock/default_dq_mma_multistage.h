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

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/dq_mma_multistage.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/warp/default_mma_tensor_op.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/warp/mma_tensorop_compute_B_with_f16.h"
#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/tile_interleaved_layout.h"

#include "paddle/phi/kernels/fusion/cutlass/cutlass_extensions/gemm/threadblock/default_dq_mma.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Type for elementA
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
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear>
struct DqMma<ElementA,
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
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >=
                                           80)>::type> {
  static_assert(platform::is_same<ElementA, half_t>::value ||
                    platform::is_same<ElementA, bfloat16_t>::value,
                "Element A must be fp16 or bf16");

  static_assert(
      platform::is_same<Operator,
                        arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
      "Mma multistage must dequantize after ldsm");

  static_assert(platform::is_same<ElementB, uint8_t>::value ||
                    platform::is_same<ElementB, uint4b_t>::value,
                "Element B must be int8 or uint4");

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  // Mma core does not depend on stages, so pass in at least 3 here to mma
  // multistage pieces are created
  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          ElementA,
                                                          LayoutA,
                                                          ElementB,
                                                          LayoutB,
                                                          ElementAccumulator,
                                                          layout::RowMajor,
                                                          OperatorClass,
                                                          std::max(kStages, 3),
                                                          Operator,
                                                          false,
                                                          CacheOpA,
                                                          CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA,
          LayoutA,
          1,
          ThreadMapA,
          AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB,
          LayoutB,
          0,
          ThreadMapB,
          AccessTypeB>;

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

  using SmemIteratorScale = IteratorScale;

  using Converter = FastInterleavedAndBiasedNumericArrayConverter<
      ElementA,
      ElementB,
      MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<
      typename MmaCore::Shape,
      IteratorA,
      typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA,
      IteratorB,
      typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB,
      IteratorScale,
      SmemIteratorScale,
      ElementAccumulator,
      layout::RowMajor,
      typename MmaCore::MmaPolicy,
      kStages,
      Converter,
      SharedMemoryClear>;
};

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
    /// Stages in GEMM
    int kStages,
    ///
    typename Operator,
    ///
    SharedMemoryClearOption SharedMemoryClear,
    ///
    int RowsPerTile,
    ///
    int ColumnsInterleaved>
struct DqMma<ElementA,
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
             kStages,
             Operator,
             SharedMemoryClear,
             typename platform::enable_if<(ArchTag::kMinComputeCapability >=
                                           80)>::type> {
  static_assert(platform::is_same<ElementA, half_t>::value ||
                    platform::is_same<ElementA, bfloat16_t>::value,
                "Element A must be fp16 or bf16");

  static_assert(
      platform::is_same<Operator,
                        arch::OpMultiplyAddDequantizeInterleavedBToA>::value,
      "Mma multistage must dequantize after ldsm");

  static_assert(platform::is_same<ElementB, uint8_t>::value ||
                    platform::is_same<ElementB, uint4b_t>::value,
                "Element B must be uint8 or uint4");

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  // Mma core does not depend on stages, so pass in at least 3 here to mma
  // multistage pieces are created
  using MmaCore =
      typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          ElementA,
                                                          LayoutA,
                                                          ElementB,
                                                          layout::ColumnMajor,
                                                          ElementAccumulator,
                                                          layout::RowMajor,
                                                          OperatorClass,
                                                          std::max(kStages, 3),
                                                          Operator,
                                                          false,
                                                          CacheOpA,
                                                          CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA,
          LayoutA,
          1,
          ThreadMapA,
          AccessTypeA>;

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
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          GmemIteratorShape,
          ElementB,
          layout::ColumnMajor,
          0,
          GmemThreadMapB,
          AccessTypeB>;

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

  using SmemIteratorScale = IteratorScale;

  using Converter = FastInterleavedAndBiasedNumericArrayConverter<
      ElementA,
      ElementB,
      MmaCore::MmaPolicy::Operator::FragmentB::kElements>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::DqMmaMultistage<
      typename MmaCore::Shape,
      IteratorA,
      typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA,
      IteratorB,
      typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB,
      IteratorScale,
      SmemIteratorScale,
      ElementAccumulator,
      layout::RowMajor,
      typename MmaCore::MmaPolicy,
      kStages,
      Converter,
      SharedMemoryClear>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
