// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/epilogue/thread/linear_combination_bias_relu.h"
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_all.h"
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.h"

namespace phi {
namespace fusion {

template <typename TShape, typename WShape, int Aligment = 1>
cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels(
    ConvAllParams params) {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using LayoutInputA = cutlass::layout::TensorNHWC;
  using LayoutInputB = cutlass::layout::TensorNHWC;
  using LayoutOutput = cutlass::layout::TensorNHWC;
  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm75;
  using ThreadblockShape = TShape;
  using WarpShape = WShape;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kFewChannels;
  using EpilogueOp =
      cutlass::epilogue::thread::LinearCombinationRelu<ElementOutput,
                                                       Aligment,
                                                       float,
                                                       ElementComputeEpilogue>;

  using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages,
      cutlass::arch::OpMultiplyAdd,
      IteratorAlgorithm,
      cutlass::conv::StrideSupport::kStrided,
      Aligment,
      Aligment>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  const half *input = params.input;
  const half *weight = params.weight;
  const half *bias = params.bias;
  half *output = params.output;
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h = params.pad_h;
  int pad_w = params.pad_w;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  const int dilationh = 1;
  const int dilationw = 1;

  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_h, pad_w, pad_w},
                                                {stride_h, stride_w},
                                                {dilationh, dilationw},
                                                {batch, oh, ow, oc},
                                                mode,
                                                1);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)(input), {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)(weight), {ic, ic * kw, ic * kw * kh}},
      {(cutlass::half_t *)(bias), {0, 0, 0}},
      {(cutlass::half_t *)(output), {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f}};

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);
  void *workspace;
  cudaMalloc(&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  cudaFree(workspace);
  return status;
}

// config 1
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 2
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 3
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<128, 32, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 4
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 5
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>>(ConvAllParams);
// config6
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>>(ConvAllParams);
// config 7
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 8
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 9
template cutlass::Status cutlass_nhwc_conv2d_bias_relu_few_channels<
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>>(ConvAllParams);

std::vector<std::function<cutlass::Status(ConvAllParams)>>
    cutlass_conv2d_bias_relu_few_channels_all_func = {
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 32, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<128, 32, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<128, 64, 64>,
            cutlass::gemm::GemmShape<32, 32, 64>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 128, 32>,
            cutlass::gemm::GemmShape<32, 64, 32>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 128, 64>,
            cutlass::gemm::GemmShape<64, 64, 32>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<64, 256, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>>,
        cutlass_nhwc_conv2d_bias_relu_few_channels<
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>>};
std::map<std::vector<int>, int> map_problem_conv2d_bias_relu_few_channels;

void cutlass_conv2d_bias_relu_few_channels(ConvAllParams params) {
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h = params.pad_h;
  int pad_w = params.pad_w;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;

  std::vector<int> problem_size = {
      batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, stride_h, stride_w};

  if (map_problem_conv2d_bias_relu_few_channels.count(problem_size)) {
    cutlass_conv2d_bias_relu_few_channels_all_func
        [map_problem_conv2d_bias_relu_few_channels.at(problem_size)](params);
    return;
  } else {
    map_problem_conv2d_bias_relu_few_channels[problem_size] = -1;
  }

  float min_time = 100000.f;
  for (int i = 0; i < cutlass_conv2d_bias_relu_few_channels_all_func.size();
       i++) {
    cutlass::Status status;
    auto func = cutlass_conv2d_bias_relu_few_channels_all_func[i];
    for (int ii = 0; ii < WARMUP; ii++) {
      status = func(params);
    }

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int ii = 0; ii < REPEAT; ii++) {
      status = func(params);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
      min_time = elapsed_time;
      map_problem_conv2d_bias_relu_few_channels[problem_size] = i;
    }

    // debug code
  }
}
}  // namespace fusion
}  // namespace phi
