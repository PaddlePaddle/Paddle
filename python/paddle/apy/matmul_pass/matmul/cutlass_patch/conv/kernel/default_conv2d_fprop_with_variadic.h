// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Defines a Conv2d fprop kernel whose epilogue is replaced by
// `EpilogueWithVariadic`, so that an arbitrary number of extra epilogue
// operands can be consumed by the output operator.

#pragma once

#include "cutlass/conv/convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/conv/kernel/implicit_gemm_convolution.h"

#include "cutlass_patch/epilogue/threadblock/default_epilogue_with_variadic.h"

namespace cutlass_patch {
namespace conv {
namespace kernel {

/// Conv2d fprop with a variadic epilogue.
///
/// The mainloop (`Mma`) is taken verbatim from the stock
/// `cutlass::conv::kernel::DefaultConv2dFprop`; only the epilogue is swapped
/// for `EpilogueWithVariadic`, which feeds the `(row, column)` offsets of every
/// output vector to the output operator. `DefaultEpilogueWithVariadicTensorOp`
/// is shared with the GEMM backend: it only depends on
/// Shape/WarpMmaOperator/PartitionsK/ElementC/OutputOp/kElementsPerAccess, none
/// of which is convolution specific.
///
/// Note that conv2d fprop always writes a packed NPQK output, hence the
/// packed `PredicatedTileIterator` picked by
/// `DefaultEpilogueWithVariadicTensorOp` is correct for any `StrideSupport`
/// (`StrideSupport` only changes the *output* iterator of dgrad).
template <
    /// Element type for the activation
    typename ElementA,
    /// Layout type for the activation
    typename LayoutA,
    /// Element type for the filter
    typename ElementB,
    /// Layout type for the filter
    typename LayoutB,
    /// Element type for C and D operands
    typename ElementC,
    /// Layout type for C and D operands
    typename LayoutC,
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
    /// Epilogue output operator - must satisfy concept of
    /// 'EpilogueWithVariadicOp'
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by the mainloop
    typename MathOperatorTag,
    /// Activation iterator algorithm
    cutlass::conv::IteratorAlgorithm IteratorAlgorithm,
    /// Stride support of the activation iterator
    cutlass::conv::StrideSupport StrideSupport,
    /// Access granularity of the activation in units of elements
    int AlignmentA,
    /// Access granularity of the filter in units of elements
    int AlignmentB>
struct DefaultConv2dFpropWithVariadic {
  /// The stock conv2d fprop kernel, used only to derive the mainloop and the
  /// shape parameters of the default epilogue.
  using ConvBase = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      ThreadblockSwizzle,
      Stages,
      MathOperatorTag,
      IteratorAlgorithm,
      StrideSupport,
      AlignmentA,
      AlignmentB>::Kernel;

  /// Replace the default epilogue by the variadic one.
  using Epilogue = typename cutlass_patch::epilogue::threadblock::
      DefaultEpilogueWithVariadicTensorOp<
          typename ConvBase::Epilogue::Shape,
          typename ConvBase::Epilogue::WarpMmaOperator,
          ConvBase::Epilogue::kPartitionsK,
          ElementC,
          EpilogueOutputOp,
          ConvBase::Epilogue::kElementsPerAccess>::Epilogue;

  /// Compose the conv2d kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
      typename ConvBase::Mma,
      Epilogue,
      ThreadblockSwizzle,
      cutlass::conv::Operator::kFprop>;
};

}  // namespace kernel
}  // namespace conv
}  // namespace cutlass_patch
