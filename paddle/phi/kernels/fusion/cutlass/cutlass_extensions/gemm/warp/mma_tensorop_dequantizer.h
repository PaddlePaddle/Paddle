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
  \brief Defines iterators used by warp-level matrix multiply operations
  targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/arch/arch.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor.h"

#include "cutlass/functional.h"
#include "cutlass/platform/platform.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Matrix multiply operator
    typename MmaOperator_,
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of Scale elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Number of threads participating in one matrix operation
    int Threads,
    ///
    /// Data type of out elements
    typename Element_out_,
    typename Enable = void>
class MmaTensorOpDequantizer;

////////////////////////////////////////////////////////////////////////////////
// Bfloat specialization for Ampere
#ifdef PADDLE_CUDA_BF16
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    bfloat16_t,
    layout::RowMajor,
    32,
    bfloat16_t,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 80 &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor =
      MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the scales
  using ElementScale = bfloat16_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementScale, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale =
      Array<ElementScale,
            kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int quad = lane_idx / 4;
    const int thread_offset = warp_offset + quad;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,
                  const FragmentScale& scale_frag) {
    // Slow path not implemented here on purpose. If we need to do HMMA on older
    // arch, scale conversion should happen before scales are stored to shared
    // memory and we should use the fp16 dequantizer. This will avoid numerous
    // conversion instructions in GEMM main loop.
    arch::device_breakpoint();
  }

 private:
  ElementScale const* pointer_;
};
#endif

// Specialization for Turing & Ampere
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    half_t,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 75 &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor =
      MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the scales
  using ElementScale = half_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementScale, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale =
      Array<ElementScale,
            kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int quad = lane_idx / 4;
    const int thread_offset = warp_offset + quad;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,
                  const FragmentScale& scale_frag) {
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB =
        Array<typename _MmaOperandB::Element,
              kExpansionFactor * _MmaOperandB::kElements>;
    static_assert(
        ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn ==
            FragmentDequantizedOperand::kElements,
        "");

    multiplies<ExpandedMmaOperandB> mul_op;

    ExpandedMmaOperandB* operand_frag_ptr =
        reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      operand_frag_ptr[mma_n_iter] =
          mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
    }
  }

 private:
  ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x RowMajor B tensorOp, for 32x32x4 interleaved
// gemm
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    half_t,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::RowMajor>::value>::type> {
 public:
  static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape,
                                  GemmShape<32, 32, 4>>::value,
                "");

  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Type of the scales
  using ElementScale = half_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementScale, MmaOperator::FragmentB::kElements>;

  /// Warp mma shape
  using Shape = Shape_;

  // Fragment to hold scale data to apply to B before mma
  // Each 32x32x4 matmul uses 8 elements from B.
  static constexpr int ColsPerMmaTile = 32;
  static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
  using FragmentScale = Array<ElementScale, TileNIterations * 8>;
  using AccessType = Array<ElementScale, 8>;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int base_col = lane_idx & 0xF8;
    const int thread_offset = warp_offset + base_col;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {  // NOLINT
    AccessType* scale_frag_ptr = reinterpret_cast<AccessType*>(&scale_frag);

    CUTLASS_PRAGMA_UNROLL
    for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
      // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
      scale_frag_ptr[tile_iter] = *reinterpret_cast<AccessType const*>(
          pointer_ + ColsPerMmaTile * tile_iter);
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,  // NOLINT
                  const FragmentScale& scale_frag) {
    static_assert(
        FragmentScale::kElements == FragmentDequantizedOperand::kElements, "");

    multiplies<FragmentDequantizedOperand> mul_op;
    operand_frag = mul_op(operand_frag, scale_frag);
  }

 private:
  ElementScale const* pointer_;
};

////////////////////////////////////////////////////////////////////////////////

// Specialization for Volta A x ColumnMajor B tensorOp, for 32x32x4 interleaved
// gemm
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    half_t,
    layout::RowMajor,
    32,
    typename platform::enable_if<
        platform::is_same<typename MmaOperator_::ArchTag, arch::Sm70>::value &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::ColumnMajor>::value>::type> {
 public:
  static_assert(platform::is_same<typename MmaOperator_::InterleavedTileShape,
                                  GemmShape<32, 32, 4>>::value,
                "");

  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Type of the scales
  using ElementScale = half_t;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementScale, MmaOperator::FragmentB::kElements>;

  /// Warp mma shape
  using Shape = Shape_;

  // Fragment to hold scale data to apply to B before mma
  // Each 32x32x4 matmul uses 8 elements from B.
  static constexpr int ColsPerMmaTile = 32;
  static constexpr int TileNIterations = Shape::kN / ColsPerMmaTile;
  using FragmentScale = Array<ElementScale, TileNIterations * 2>;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int base_col = lane_idx & 0xF8 + lane_idx % 4;
    const int thread_offset = warp_offset + base_col;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {  // NOLINT
    CUTLASS_PRAGMA_UNROLL
    for (int tile_iter = 0; tile_iter < TileNIterations; ++tile_iter) {
      // We jump by 32 here since volta does <32x32x4> super mmas inside a warp.
      // For col major B, each thread will jump 4 cols to get its next value
      // inside of the super mma.
      CUTLASS_PRAGMA_UNROLL
      for (int mma_iter = 0; mma_iter < 2; ++mma_iter) {
        scale_frag[tile_iter * 2 + mma_iter] =
            pointer_[ColsPerMmaTile * tile_iter + 4 * mma_iter];
      }
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,  // NOLINT
                  const FragmentScale& scale_frag) {
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    static constexpr int total_n_mmas = 2 * TileNIterations;
    static_assert(MmaOperandB::kElements * total_n_mmas ==
                      FragmentDequantizedOperand::kElements,
                  "");

    multiplies<MmaOperandB> mul_op;

    MmaOperandB* operand_frag_ptr =
        reinterpret_cast<MmaOperandB*>(&operand_frag);
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < total_n_mmas; ++mma_n_iter) {
      operand_frag_ptr[mma_n_iter] =
          mul_op(operand_frag_ptr[mma_n_iter], scale_frag[mma_n_iter]);
    }
  }

 private:
  ElementScale const* pointer_;
};

// Specialization for Turing & Ampere when Scale type is float and output type
// is half_t.
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    float,
    layout::RowMajor,
    32,
    half_t,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 75 &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor =
      MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the output
  using ElementType = half_t;

  // using ElementScale = float;
  using ElementScale = float;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementType, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale =
      Array<ElementScale,
            kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int quad = lane_idx / 4;
    const int thread_offset = warp_offset + quad;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,
                  const FragmentScale& scale_frag) {
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB =
        Array<typename _MmaOperandB::Element,
              kExpansionFactor * _MmaOperandB::kElements>;

    using ComputeFrag =
        Array<ElementScale, kExpansionFactor * _MmaOperandB::kElements>;

    static_assert(
        ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn ==
            FragmentDequantizedOperand::kElements,
        "");

    multiplies<ComputeFrag> mul_op;

    ExpandedMmaOperandB* operand_frag_ptr =
        reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

    NumericArrayConverter<ElementScale,
                          ElementType,
                          kExpansionFactor * _MmaOperandB::kElements>
        source_converter;
    NumericArrayConverter<ElementType,
                          ElementScale,
                          kExpansionFactor * _MmaOperandB::kElements>
        output_converter;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      ComputeFrag convert_frag = source_converter(operand_frag_ptr[mma_n_iter]);
      convert_frag = mul_op(convert_frag, scale_frag[mma_n_iter]);
      operand_frag_ptr[mma_n_iter] = output_converter(convert_frag);
    }
  }

 private:
  ElementScale const* pointer_;
};

// Specialization for Turing & Ampere when Scale type is float and output type
// is bfloat16.
#ifdef PADDLE_CUDA_BF16
template <
    /// Underlying matrix multiply operator (concept: MmaTensorOp)
    typename MmaOperator_,
    /// Shape of the warp level matrix multiply (concept: GemmShape)
    typename Shape_>
class MmaTensorOpDequantizer<
    MmaOperator_,
    Shape_,
    Operand::kB,
    float,
    layout::RowMajor,
    32,
    bfloat16_t,
    typename platform::enable_if<
        MmaOperator_::ArchTag::kMinComputeCapability >= 75 &&
        platform::is_same<typename MmaOperator_::ArchMmaOperator::LayoutB,
                          layout::ColumnMajor>::value>::type> {
 public:
  /// Mma Operator
  using MmaOperator = MmaOperator_;

  // The architecture specific mma ooperator being used
  using ArchMmaOperator = typename MmaOperator::ArchMmaOperator;

  // Mma Instruction Shape
  using InstructionShape = typename ArchMmaOperator::Shape;

  // This is the ratio of the load instruction vs the compute instruction.
  static constexpr int kExpansionFactor =
      MmaOperator::IteratorB::InstructionShape::kRow / InstructionShape::kK;

  /// Type of the output
  using ElementType = bfloat16_t;

  // using ElementScale = float;
  using ElementScale = float;

  /// Fragment to hold B data before Mma
  using FragmentDequantizedOperand =
      Array<ElementType, MmaOperator::FragmentB::kElements>;

  // Fragment to hold scale data to apply to B before mma
  // We need 1 fp16 per matrix iteration in the N dimension
  static constexpr int kColsPerMmaPerThread = 1;
  using FragmentScale =
      Array<ElementScale,
            kColsPerMmaPerThread * MmaOperator::MmaIterations::kColumn>;

  /// Warp mma shape
  using Shape = Shape_;

  /// Layout of the scales in shared memory
  using Layout = layout::RowMajor;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<ElementScale, Layout>;

  CUTLASS_DEVICE
  MmaTensorOpDequantizer(TensorRef smem_scales,
                         const int warp_idx_n,
                         const int lane_idx) {
    const int warp_offset = warp_idx_n * Shape::kN;
    const int quad = lane_idx / 4;
    const int thread_offset = warp_offset + quad;
    pointer_ = smem_scales.data() + thread_offset;
  }

  CUTLASS_DEVICE
  void load(FragmentScale& scale_frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      scale_frag[mma_n_iter] = pointer_[mma_n_iter * InstructionShape::kN];
    }
  }

  CUTLASS_DEVICE
  void dequantize(FragmentDequantizedOperand& operand_frag,
                  const FragmentScale& scale_frag) {
    using _MmaOperandB = typename ArchMmaOperator::FragmentB;
    using ExpandedMmaOperandB =
        Array<typename _MmaOperandB::Element,
              kExpansionFactor * _MmaOperandB::kElements>;

    using ComputeFrag =
        Array<ElementScale, kExpansionFactor * _MmaOperandB::kElements>;

    static_assert(
        ExpandedMmaOperandB::kElements * MmaOperator::MmaIterations::kColumn ==
            FragmentDequantizedOperand::kElements,
        "");

    multiplies<ComputeFrag> mul_op;

    ExpandedMmaOperandB* operand_frag_ptr =
        reinterpret_cast<ExpandedMmaOperandB*>(&operand_frag);

    NumericArrayConverter<ElementScale,
                          ElementType,
                          kExpansionFactor * _MmaOperandB::kElements>
        source_converter;
    NumericArrayConverter<ElementType,
                          ElementScale,
                          kExpansionFactor * _MmaOperandB::kElements>
        output_converter;

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n_iter = 0; mma_n_iter < MmaOperator::MmaIterations::kColumn;
         ++mma_n_iter) {
      ComputeFrag convert_frag = source_converter(operand_frag_ptr[mma_n_iter]);
      convert_frag = mul_op(convert_frag, scale_frag[mma_n_iter]);
      operand_frag_ptr[mma_n_iter] = output_converter(convert_frag);
    }
  }

 private:
  ElementScale const* pointer_;
};
#endif
////////////////////////////////////////////////////////////////////////////////

}  // namespace warp
}  // namespace gemm
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
