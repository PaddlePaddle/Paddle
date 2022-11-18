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
#include "conv2d_util.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/conv/device/implicit_gemm_convolution.h"
#include "cutlass/conv/kernel/default_conv2d_fprop.h"
#include "cutlass/cutlass.h"


template <typename TShape, typename WShape, int aligment = 8>
cutlass::Status cutlass_nhwc_conv2d_bias_silu(COMMON_CONV_PARAMS) {
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
      cutlass::conv::IteratorAlgorithm::kOptimized;
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationSilu<
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
      aligment,
      aligment>::Kernel;
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
  cudaFree(workspace);
  return status;
}

// config 1
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>>
(COMMON_CONV_PARAMS);
// config 2
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<32, 32, 64>> 
(COMMON_CONV_PARAMS);
// config 3
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 32, 64>, cutlass::gemm::GemmShape<32, 32, 64>>
(COMMON_CONV_PARAMS);
// config 4
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>>
(COMMON_CONV_PARAMS);
// config 5
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>> 
(COMMON_CONV_PARAMS);
// config6
template cutlass::Status 
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>>
(COMMON_CONV_PARAMS);
// config 7
template cutlass::Status
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 64, 32>>
(COMMON_CONV_PARAMS);
// config 8
template cutlass::Status
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 256, 32>, cutlass::gemm::GemmShape<64, 64, 32>>
(COMMON_CONV_PARAMS);
// config 9
template cutlass::Status
cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>>
(COMMON_CONV_PARAMS);

#define N 9
cutlass::Status (*cutlass_conv2d_bias_silu_all_func[N])(const half *,
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
                                       int) = {
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 32, 64>, cutlass::gemm::GemmShape<32, 32, 64>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 32, 64>, cutlass::gemm::GemmShape<32, 32, 64>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 64, 64>, cutlass::gemm::GemmShape<32, 32, 64>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 128, 32>, cutlass::gemm::GemmShape<32, 64, 32>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 128, 64>, cutlass::gemm::GemmShape<64, 64, 32>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<64, 256, 32>, cutlass::gemm::GemmShape<64, 64, 32>>,
                                                cutlass_nhwc_conv2d_bias_silu <cutlass::gemm::GemmShape<128, 64, 32>, cutlass::gemm::GemmShape<64, 32, 32>>
                                               };
std::map<std::vector<int>,   int> map_problem_conv2d_bias_silu;



void cutlass_conv2d_bias_silu(COMMON_CONV_PARAMS) {
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

 if (map_problem_conv2d_bias_silu.count(problem_size)) {
    cutlass_conv2d_bias_silu_all_func[map_problem_conv2d_bias_silu.at(problem_size)](COMMON_CONV_ARGS);
    return;
 }
 else {
   map_problem_conv2d_bias_silu[problem_size] = -1;
 }
  
  float min_time = 100000.f;
  for (int i = 0; i < N; i++) {
    cutlass::Status status;
    auto func = cutlass_conv2d_bias_silu_all_func[i];
    for (int i = 0; i < WARMUP; i++) {
      status = func(COMMON_CONV_ARGS);
    }

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg);
    for (int i = 0; i < REPEATE; i++) {
      status = func(COMMON_CONV_ARGS);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, beg, end);
    if (elapsed_time < min_time && status == cutlass::Status::kSuccess) {
        min_time = elapsed_time;
        map_problem_conv2d_bias_silu[problem_size] = i;
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
    // std::cout << "max diff : "  <<  diff(output_from_cutlass, cpu_output, output_size) << std::endl;
    // free(cpu_output);
    // free(cpu_input);
    // free(cpu_weight);
    // free(cpu_bias);
    // free(output_from_cutlass);
  }

}
