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

#pragma once

#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_ref.h"

#include "cutlass_patch/cuda/cutlass_common.cuh"
#include "cutlass_patch/cuda/default_config_id.h"

#include "params.h"  // NOLINT

namespace ap {

// Conv2d forward propagation fused with a per-channel bias add and relu:
//   output = relu(conv2d(input, weight) + bias)
//
// Implemented as an implicit GEMM with the `kOptimized` activation iterator and
// the stock cutlass `LinearCombinationRelu` epilogue.
template <typename ElementT,
          typename ElementComputeT,
          int AlignA = 128 / cutlass::sizeof_bits<ElementT>::value,
          int AlignB = 128 / cutlass::sizeof_bits<ElementT>::value,
          int AlignC = 128 / cutlass::sizeof_bits<ElementT>::value,
          int ConfigId = DefaultConfig::kConfigId,
          int SwizzleFactor = DefaultConfig::kSwizzleFactor>
void Conv2dAddRelu(const Conv2dEpilogueParams &params) {
  using ElementAccumulator =
      typename CutlassDataType<ElementComputeT>::Type;  // <- data type of
                                                        // accumulator
  using ElementComputeEpilogue =
      ElementAccumulator;  // <- data type of epilogue operations
  using ElementInputA =
      typename CutlassDataType<ElementT>::Type;  // <- data type of activation
  using ElementInputB =
      typename CutlassDataType<ElementT>::Type;  // <- data type of filter
  using ElementOutput =
      typename CutlassDataType<ElementT>::Type;  // <- data type of output

  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;

  // The threadblock/warp tile table is shared with matmul: conv2d is lowered to
  // an implicit GEMM, so the same tiling applies.
  using TuningConfig =
      GemmTuningConfigs<ElementT, SwizzleFactor, /*Batched=*/false, ConfigId>;

  // relu(alpha * accumulator + beta * source), `source` is the per-channel bias
  // broadcast along N/P/Q by a zero-strided TensorRef.
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      AlignC,
      ElementAccumulator,
      ElementComputeEpilogue,
      cutlass::epilogue::thread::ScaleType::Default>;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      typename TuningConfig::TShape,
      typename TuningConfig::WShape,
      typename TuningConfig::IShape,
      EpilogueOutputOp,
      typename TuningConfig::SwizzleThreadBlock,
      TuningConfig::kNumStages,
      typename GemmOperation<ElementT>::Type,
      cutlass::conv::IteratorAlgorithm::kOptimized,
      cutlass::conv::StrideSupport::kStrided,
      AlignA,
      AlignB>::Kernel;

  using ImplicitGemmFunc =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  cutlass::conv::Conv2dProblemSize problem_size{
      cutlass::Tensor4DCoord{params.N, params.H, params.W, params.C},
      cutlass::Tensor4DCoord{params.K, params.R, params.S, params.C},
      cutlass::Tensor4DCoord{
          params.pad_h, params.pad_h, params.pad_w, params.pad_w},
      cutlass::MatrixCoord{params.stride_h, params.stride_w},
      cutlass::MatrixCoord{params.dilation_h, params.dilation_w},
      cutlass::Tensor4DCoord{params.N, params.P, params.Q, params.K},
      cutlass::conv::Mode::kCrossCorrelation,
      /*split_k_slices=*/1,
      params.groups};

  // cutlass takes mutable TensorRef for all the operands.
  ElementInputA *input = const_cast<ElementInputA *>(
      reinterpret_cast<const ElementInputA *>(params.input));
  ElementInputB *weight = const_cast<ElementInputB *>(
      reinterpret_cast<const ElementInputB *>(params.weight));
  ElementOutput *bias = const_cast<ElementOutput *>(
      reinterpret_cast<const ElementOutput *>(params.bias));
  ElementOutput *output = reinterpret_cast<ElementOutput *>(params.output);

  ElementComputeEpilogue alpha = static_cast<ElementComputeEpilogue>(1);
  ElementComputeEpilogue beta = bias ? static_cast<ElementComputeEpilogue>(1)
                                     : static_cast<ElementComputeEpilogue>(0);

  typename ImplicitGemmFunc::UnderlyingKernel::TensorRefA ref_A{
      input, LayoutInputA::packed(problem_size.activation_extent())};
  typename ImplicitGemmFunc::UnderlyingKernel::TensorRefB ref_B{
      weight, LayoutInputB::packed(problem_size.filter_extent())};
  // Zero stride projects away the N, H, W dimensions of the bias.
  typename ImplicitGemmFunc::UnderlyingKernel::TensorRefC ref_C{
      bias, LayoutOutput::Stride(0)};
  typename ImplicitGemmFunc::UnderlyingKernel::TensorRefC ref_D{
      output, LayoutOutput::packed(problem_size.output_extent())};

  typename ImplicitGemmFunc::Arguments arguments{
      problem_size,
      ref_A,           // <- input, activation, shape={N, H, W, C}
      ref_B,           // <- input, filter, shape={K, R, S, C}
      ref_C,           // <- input, bias, shape={K}
      ref_D,           // <- output, shape={N, P, Q, K}
      {alpha, beta}};  // <- epilogue params, alpha, beta

  size_t workspace_size = ImplicitGemmFunc::get_workspace_size(arguments);
  void *workspace = workspace_size > 0 ? GetWorkspace(workspace_size) : nullptr;

  ImplicitGemmFunc implicit_gemm;

  cudaStream_t *stream_ptr =
      reinterpret_cast<cudaStream_t *>(params.stream_ptr);

  CHECK_CUTLASS(implicit_gemm.can_implement(arguments));
  CHECK_CUTLASS(implicit_gemm.initialize(arguments, workspace, *stream_ptr));

  //
  // Run the implicit GEMM
  //
  CHECK_CUTLASS(implicit_gemm(*stream_ptr));
#if AP_ENABLE_DEBUG
  CHECK_CUDA(cudaStreamSynchronize(*stream_ptr));
#endif
}

}  // namespace ap
