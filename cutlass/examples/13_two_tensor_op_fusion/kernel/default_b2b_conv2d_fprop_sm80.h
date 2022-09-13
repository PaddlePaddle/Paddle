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

/*! \file
    \brief 
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped 
      matrix multiply-add with the appropriate threadblock-scoped epilogue.  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"

#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/gemm/warp/mma_tensor_op_fragment_iterator.h"

#include "kernel/default_b2b_conv2d_fprop.h"
#include "kernel/b2b_implicit_gemm_convolution.h"
#include "threadblock/b2b_implicit_gemm_multistage.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp convolutions 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage 
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape0,
  typename ThreadblockShape1,
  typename WarpShape0,
  typename WarpShape1,
  typename InstructionShape,
  typename EpilogueOutputOp0,
  typename EpilogueOutputOp1,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultB2bConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic
> {

  // Define the core components from GEMM
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA0 = typename MmaCore0::IteratorThreadMapA;
  using IteratorA0 =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
      ElementA, LayoutA,
      ThreadMapA0
    >;

  using SmemIteratorA0 = typename MmaCore0::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB0 = typename MmaCore0::IteratorThreadMapB;
  using IteratorB0 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
      ElementB, LayoutB,
      ThreadMapB0
    >;
  
  using SmemIteratorB0 = typename MmaCore0::SmemIteratorB;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::ColumnMajor;
  using FragmentIteratorA1 = 
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp0>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  // Define iterators over tiles from the B operand
  using ThreadMapB1 = typename MmaCore1::IteratorThreadMapB;
  using IteratorB1 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
      ElementB, LayoutB,
      ThreadMapB1
    >;
  
  using SmemIteratorB1 = typename MmaCore1::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistage<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    ThreadblockShape1,
    FragmentIteratorA1,
    IteratorAccumulatorScaleBias,
    FragmentIteratorA1ScaleBias,
    IteratorB1,
    SmemIteratorB1,
    arch::CacheOperation::Global,
    EpilogueOutputOp0,
    MmaPolicy0,
    MmaPolicy1,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape1,
    WarpMmaTensorOp1,
    1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::B2bImplicitGemmConvolution<
    B2bMma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape0,
  typename ThreadblockShape1,
  typename WarpShape0,
  typename WarpShape1,
  typename InstructionShape,
  typename EpilogueOutputOp0,
  typename EpilogueOutputOp1,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  int InterleavedK
>
struct DefaultB2bConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic
> {

  // Define the core components from GEMM
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, MathOperatorTag, true>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA0 = typename MmaCore0::SmemThreadMapA;
  using IteratorA0 =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
      ElementA, layout::TensorNCxHWx<InterleavedK>,
      ThreadMapA0
    >;

  using SmemIteratorA0 = typename MmaCore0::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB0 = typename MmaCore0::SmemThreadMapB;
  using IteratorB0 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
      ElementB, layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB0
    >;
  
  using SmemIteratorB0 = typename MmaCore0::SmemIteratorB;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::RowMajor;
  using FragmentIteratorA1 = 
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp0>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 4;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>, 
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>, 
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  using ThreadMapB1 = typename MmaCore1::SmemThreadMapB;
  using IteratorB1 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
      ElementB, layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB1
    >;
 
  using SmemIteratorB1 = typename MmaCore1::SmemIteratorB;


  // Warp-level GEMM components
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistage<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    ThreadblockShape1,
    FragmentIteratorA1,
    IteratorAccumulatorScaleBias,
    FragmentIteratorA1ScaleBias,
    IteratorB1,
    SmemIteratorB1,
    arch::CacheOperation::Global,
    EpilogueOutputOp0,
    MmaPolicy0,
    MmaPolicy1,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape1,
    WarpMmaTensorOp1,
    1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::B2bImplicitGemmConvolution<
    B2bMma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and 
/// multistage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape0,
  typename ThreadblockShape1,
  typename WarpShape0,
  typename WarpShape1,
  typename InstructionShape,
  typename EpilogueOutputOp0,
  typename EpilogueOutputOp1,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag
>
struct DefaultB2bConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {

  // Define the core components from GEMM
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA0 = typename MmaCore0::IteratorThreadMapA;
  using IteratorA0 =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
      ElementA, LayoutA,
      ThreadMapA0
    >;

  using SmemIteratorA0 = typename MmaCore0::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB0 = typename MmaCore0::IteratorThreadMapB;
  using IteratorB0 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
      ElementB, LayoutB,
      ThreadMapB0
    >;
  
  using SmemIteratorB0 = typename MmaCore0::SmemIteratorB;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::ColumnMajor;
  using FragmentIteratorA1 = 
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp0>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 2;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>,
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>,
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  // Define iterators over tiles from the B operand
  using ThreadMapB1 = typename MmaCore1::IteratorThreadMapB;
  using IteratorB1 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
      ElementB, LayoutB,
      ThreadMapB1
    >;
  
  using SmemIteratorB1 = typename MmaCore1::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistage<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    ThreadblockShape1,
    FragmentIteratorA1,
    IteratorAccumulatorScaleBias,
    FragmentIteratorA1ScaleBias,
    IteratorB1,
    SmemIteratorB1,
    arch::CacheOperation::Global,
    EpilogueOutputOp0,
    MmaPolicy0,
    MmaPolicy1,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape1,
    WarpMmaTensorOp1,
    1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::B2bImplicitGemmConvolution<
    B2bMma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimzed IteratorAlgorithm and 
// multistage pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape0,
  typename ThreadblockShape1,
  typename WarpShape0,
  typename WarpShape1,
  typename InstructionShape,
  typename EpilogueOutputOp0,
  typename EpilogueOutputOp1,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  int InterleavedK
>
struct DefaultB2bConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape0,
  ThreadblockShape1,
  WarpShape0,
  WarpShape1,
  InstructionShape,
  EpilogueOutputOp0,
  EpilogueOutputOp1,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized
> {

  // Define the core components from GEMM
  using MmaCore0 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape0, WarpShape0, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, MathOperatorTag, true>;
  using MmaCore1 = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape1, WarpShape1, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA0 = typename MmaCore0::SmemThreadMapA;
  using IteratorA0 =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kK>,
      ElementA, layout::TensorNCxHWx<InterleavedK>,
      ThreadMapA0
    >;

  using SmemIteratorA0 = typename MmaCore0::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB0 = typename MmaCore0::SmemThreadMapB;
  using IteratorB0 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape0::kK, ThreadblockShape0::kN>,
      ElementB, layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB0
    >;
  
  using SmemIteratorB0 = typename MmaCore0::SmemIteratorB;

  // Use fragment iterator for A operand
  using AccumulatorLayout = cutlass::layout::RowMajor;
  using FragmentIteratorA1 = 
      cutlass::gemm::warp::MmaTensorOpFragmentIterator<
          cutlass::MatrixShape<MmaCore1::WarpShape::kM, MmaCore1::InstructionShape::kK>, //warp shape
          cutlass::MatrixShape<MmaCore0::WarpShape::kM, MmaCore0::WarpShape::kN>, //accumulator shape
          MmaCore1::Shape::kK, //kBlocksColumn
          ElementAccumulator, ElementA, AccumulatorLayout, InstructionShape, EpilogueOutputOp0>;

  /// Define iterators over tiles from scale/bias vectors
  using ElementScaleBias = typename EpilogueOutputOp0::ElementCompute;
  using LayoutScaleBias = layout::RowMajor; //vector layout doesn't really matter
  static int const kElementsPerAccess = 4;
  using IteratorAccumulatorScaleBias =
    cutlass::transform::threadblock::VectorIterator<
      cutlass::transform::threadblock::PredicatedVectorAccessIterator<
          cutlass::MatrixShape<ThreadblockShape0::kM, ThreadblockShape0::kN>, 
          cutlass::MatrixShape<WarpShape1::kM, WarpShape1::kK>, 
          ElementScaleBias, LayoutScaleBias, kElementsPerAccess>
    >;

  // Warp-level iterators to load scale and bias vectors
  using FragmentIteratorA1ScaleBias = cutlass::transform::warp::VectorFragmentIterator<
      MatrixShape<1, IteratorAccumulatorScaleBias::Fragment::kElements>, ElementScaleBias,
      LayoutScaleBias, InstructionShape, kElementsPerAccess>;

  using ThreadMapB1 = typename MmaCore1::SmemThreadMapB;
  using IteratorB1 =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape1::kK, ThreadblockShape1::kN>,
      ElementB, layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB1
    >;
 
  using SmemIteratorB1 = typename MmaCore1::SmemIteratorB;


  // Warp-level GEMM components
  using WarpMmaTensorOp1 = typename MmaCore1::MmaTensorOp;
  using MmaPolicy0 = typename MmaCore0::MmaPolicy;
  using MmaPolicy1 = typename MmaCore1::MmaPolicy;

  // Define the Mma
  using B2bMma = threadblock::B2bImplicitGemmMultistage<
    ThreadblockShape0,
    IteratorA0,
    SmemIteratorA0,
    arch::CacheOperation::Always,
    IteratorB0,
    SmemIteratorB0,
    arch::CacheOperation::Global,
    ThreadblockShape1,
    FragmentIteratorA1,
    IteratorAccumulatorScaleBias,
    FragmentIteratorA1ScaleBias,
    IteratorB1,
    SmemIteratorB1,
    arch::CacheOperation::Global,
    EpilogueOutputOp0,
    MmaPolicy0,
    MmaPolicy1,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape1,
    WarpMmaTensorOp1,
    1,
    EpilogueOutputOp1,
    EpilogueOutputOp1::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::B2bImplicitGemmConvolution<
    B2bMma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
