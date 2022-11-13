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
#include "cutlass/gemm/device/gemm.h"
#include <cutlass/epilogue/thread/linear_combination_bias_relu.h>
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"


#define CONV_PARAMS0                                                         \
  input, weight, bias, output, batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, \
      stride_h, stride_w



cutlass::Status cutlass_nhwc_conv2d_bias_relu1(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}

cutlass::Status cutlass_nhwc_conv2d_bias_relu2(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}

cutlass::Status cutlass_nhwc_conv2d_bias_relu3(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 32, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}

cutlass::Status cutlass_nhwc_conv2d_bias_relu4(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<128, 64, 64>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}


cutlass::Status cutlass_nhwc_conv2d_bias_relu5(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}


cutlass::Status cutlass_nhwc_conv2d_bias_relu6(const half *input,
                                   const half *weight,
                                   const half *bias,
                                   half *output,
                                   int batch,
                                   int ic,
                                   int ih,
                                   int iw,
                                   int kh,
                                   int kw,
                                   int oc,
                                   int pad_h,
                                   int pad_w,
                                   int stride_h,
                                   int stride_w) {
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
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 32>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 32>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>;
  constexpr int NumStages = 2;
  static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationRelu<
      ElementOutput,
      128 / cutlass::sizeof_bits<ElementOutput>::value,
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
      8,
      8>::Kernel;
  using ImplicitGemm =
      cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

  int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
  int ow = (iw + pad_w * 2 - kw) / stride_w + 1;
  cutlass::conv::Mode mode = cutlass::conv::Mode::kCrossCorrelation;
  cutlass::conv::Conv2dProblemSize problem_size({batch, ih, iw, ic},
                                                {oc, kh, kw, ic},
                                                {pad_h, pad_w, pad_h, pad_w},
                                                {stride_h, stride_w},
                                                {1, 1},
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
  cudaMalloc((void **)&workspace, bytes);

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  check(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  check(status);
  status = implicit_gemm_op();
  check(status);
  return status;
}


#define N 6
cutlass::Status (*cutlass_conv2d_bias_relu_all_func[N])(const half *,
                                       const half *,
                                       const half *,
                                       half *,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int,
                                       int) = {cutlass_nhwc_conv2d_bias_relu1,
                                               cutlass_nhwc_conv2d_bias_relu2,
                                               cutlass_nhwc_conv2d_bias_relu3, 
                                               cutlass_nhwc_conv2d_bias_relu4,
                                               cutlass_nhwc_conv2d_bias_relu5,
                                               cutlass_nhwc_conv2d_bias_relu6
                                               };
std::map<std::vector<int>,   int> map_problem2_func0;

void cutlass_nhwc_conv2d_bias_relu(const half *input,
                       const half *weight,
                       const half *bias,
                       half *output,
                       int batch,
                       int ic,
                       int ih,
                       int iw,
                       int kh,
                       int kw,
                       int oc,
                       int pad_h,
                       int pad_w,
                       int stride_h,
                       int stride_w) {
  std::vector<int> problem_size;
  problem_size.push_back(batch);
  problem_size.push_back(ic);
  problem_size.push_back(ih);
  problem_size.push_back(iw);
  problem_size.push_back(kh);
  problem_size.push_back(kw);
  problem_size.push_back(oc);
  problem_size.push_back(pad_h);
  problem_size.push_back(pad_w);
  problem_size.push_back(stride_h);
  problem_size.push_back(stride_w);

 if (map_problem2_func0.count(problem_size)) {
     std::cout << map_problem2_func0[problem_size] << std::endl;
    cutlass_conv2d_bias_relu_all_func[map_problem2_func0.at(problem_size)](CONV_PARAMS0);
    return;
 }
  
  float min_time = 100000.f;
  for (int i = 0; i < N; i++) {
    auto func = cutlass_conv2d_bias_relu_all_func[i];
    for (int i = 0; i < WARMUP; i++) {
      func(CONV_PARAMS0);
    }

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int i = 0; i < REPEATE; i++) {
      func(CONV_PARAMS0);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    printf("gpu conv compute time: %f\n", elapsed_time);
    if (elapsed_time < min_time) {
        min_time = elapsed_time;
        map_problem2_func0[problem_size] = i;
    }

    // debug code
    // half *cpu_input, *cpu_weight, *cpu_bias;
    // float *cpu_output;
    // half *output_from_cutlass;
    
    // int input_size = batch * ic * ih * iw;
    // int weight_size = oc * ic * kh * kw;
    // cpu_input = (half*)malloc(sizeof(half) * input_size);
    // cpu_weight = (half*) malloc(sizeof(half) * weight_size);
    // cpu_bias = (half*) malloc(sizeof(half) * oc);
    // cudaMemcpy(cpu_input, input, input_size * sizeof(half), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpu_weight, weight, weight_size * sizeof(half), cudaMemcpyDeviceToHost);
    // cudaMemcpy(cpu_bias, bias, oc * sizeof(half), cudaMemcpyDeviceToHost);

    // int oh = (ih + pad_h * 2 - kh) / stride_h + 1;
    // int ow = (iw + pad_w * 2 - kw) / stride_w + 1;

    // int output_size = batch * oc * oh * ow;
    // cpu_output = (float*) malloc(sizeof(float) * output_size);
    // output_from_cutlass = (half*)malloc(sizeof(half) * output_size); 
    // cudaMemcpy(output_from_cutlass, output, output_size * sizeof(half), cudaMemcpyDeviceToHost);

    // naive_conv_cpu(cpu_input, cpu_weight, cpu_bias, cpu_output, batch, ic, ih, iw, kh, kw, oc, pad_h, pad_w, \
    //   stride_h, stride_w, nullptr);
    // //std::cout << "max diff : "  <<  diff(output_from_cutlass, cpu_output, output_size) << std::endl;
    // free(cpu_output);
    // free(cpu_input);
    // free(cpu_weight);
    // free(cpu_bias);
    // free(output_from_cutlass);
  }
}
