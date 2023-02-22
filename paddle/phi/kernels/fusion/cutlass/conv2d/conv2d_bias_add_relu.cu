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

#include <mutex>
#include "cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h"
#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/fusion/cutlass/conv2d/conv2d_util.h"

namespace phi {
namespace fusion {

namespace cutlass_internal {

template <typename TShape, typename WShape, int Alignment = 8>
cutlass::Status Conv2dBiasAddReluImpl(ConvAllParams params) {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidualBlock<
      cutlass::half_t,
      float,
      float,
      cutlass::half_t,
      Alignment,
      cutlass::epilogue::thread::Identity,
      cutlass::plus,
      cutlass::epilogue::thread::ReLu>;

  using Conv2dFpropKernel =
      typename cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
          cutlass::half_t,
          cutlass::layout::TensorNHWC,
          cutlass::half_t,
          cutlass::layout::TensorNHWC,
          cutlass::half_t,
          cutlass::layout::TensorNHWC,
          float,
          cutlass::arch::OpClassTensorOp,
          cutlass::arch::Sm75,
          TShape,
          WShape,
          cutlass::gemm::GemmShape<16, 8, 8>,
          EpilogueOp,
          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>,
          2,
          cutlass::arch::OpMultiplyAdd,
          cutlass::conv::IteratorAlgorithm::kOptimized,
          cutlass::conv::StrideSupport::kStrided,
          Alignment,
          Alignment>::Kernel;

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
  int pad_h0 = params.pad_h0;
  int pad_w0 = params.pad_w0;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;
  const half *residual = params.residual;

  int oh = params.oh;
  int ow = params.ow;
  int dilation_h = params.dilation_h;
  int dilation_w = params.dilation_w;

  cutlass::conv::Conv2dProblemSize problem_size(
      {batch, ih, iw, ic},
      {oc, kh, kw, ic},
      {pad_h0, 0, pad_w0, 0},
      {stride_h, stride_w},
      {dilation_h, dilation_w},
      {batch, oh, ow, oc},
      cutlass::conv::Mode::kCrossCorrelation,
      1);

  typename ImplicitGemm::Arguments arguments{
      problem_size,
      {(cutlass::half_t *)input, {ic, ic * iw, ic * iw * ih}},
      {(cutlass::half_t *)weight, {ic, ic * kw, ic * kw * kh}},
      {(cutlass::half_t *)residual, {oc, oc * ow, oc * ow * oh}},
      {(cutlass::half_t *)output, {oc, oc * ow, oc * ow * oh}},
      {1.f, 1.f},
      cutlass::conv::SplitKMode::kSerial,
      (cutlass::half_t *)(bias),
      nullptr,
      0,
      oc};

  ImplicitGemm implicit_gemm_op;
  size_t bytes = implicit_gemm_op.get_workspace_size(arguments);

  auto ctx = params.ctx;
  auto stream = ctx->stream();
  phi::Allocator::AllocationPtr tmp_gpu_ptrs_data = phi::memory_utils::Alloc(
      ctx->GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(stream)));
  void *workspace = tmp_gpu_ptrs_data->ptr();

  cutlass::Status status = implicit_gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op.initialize(arguments, workspace);
  CUTLASS_CHECK(status);
  status = implicit_gemm_op(stream);
  CUTLASS_CHECK(status);
  return status;
}

// config 0
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 64, 64>,
                          cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 1
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 32, 64>,
                          cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 2
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 32, 64>,
                          cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 3
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 64, 64>,
                          cutlass::gemm::GemmShape<32, 32, 64>>(ConvAllParams);
// config 4
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 64, 32>,
                          cutlass::gemm::GemmShape<32, 32, 32>>(ConvAllParams);
// config 5
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 128, 32>,
                          cutlass::gemm::GemmShape<32, 64, 32>>(ConvAllParams);
// config 6
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 128, 64>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 7
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 256, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 8
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 64, 32>,
                          cutlass::gemm::GemmShape<64, 32, 32>>(ConvAllParams);
// config 9
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 128, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 10
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 256, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 11
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<256, 64, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);
// config 12
template cutlass::Status
    Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<256, 128, 32>,
                          cutlass::gemm::GemmShape<64, 64, 32>>(ConvAllParams);

std::vector<std::function<cutlass::Status(ConvAllParams)>>
    conv2d_bias_add_relu_all_func = {
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 64, 32>,
                              cutlass::gemm::GemmShape<32, 32, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 64, 64>,
                              cutlass::gemm::GemmShape<32, 32, 64>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 32, 64>,
                              cutlass::gemm::GemmShape<32, 32, 64>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 32, 64>,
                              cutlass::gemm::GemmShape<32, 32, 64>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 64, 64>,
                              cutlass::gemm::GemmShape<32, 32, 64>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 128, 32>,
                              cutlass::gemm::GemmShape<32, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 128, 64>,
                              cutlass::gemm::GemmShape<64, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<64, 256, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 64, 32>,
                              cutlass::gemm::GemmShape<64, 32, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 128, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<128, 256, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<256, 64, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>>,
        Conv2dBiasAddReluImpl<cutlass::gemm::GemmShape<256, 128, 32>,
                              cutlass::gemm::GemmShape<64, 64, 32>>};
std::map<std::vector<int>, int> map_problem_conv2d_bias_add_relu;
std::mutex conv2d_bias_add_relu_mutex;

void Conv2dBiasAddRelu(ConvAllParams params) {
  int batch = params.batch;
  int ic = params.ic;
  int ih = params.ih;
  int iw = params.iw;
  int kh = params.kh;
  int kw = params.kw;
  int oc = params.oc;
  int pad_h0 = params.pad_h0;
  int pad_w0 = params.pad_w0;
  int stride_h = params.stride_h;
  int stride_w = params.stride_w;

  std::vector<int> problem_size = {
      batch, ic, ih, iw, kh, kw, oc, pad_h0, pad_w0, stride_h, stride_w};

  if (map_problem_conv2d_bias_add_relu.count(problem_size)) {
    conv2d_bias_add_relu_all_func[map_problem_conv2d_bias_add_relu.at(
        problem_size)](params);
    return;
  }

  std::lock_guard<std::mutex> guard(conv2d_bias_add_relu_mutex);

  // config 6's diff is large.
  conv2d_bias_add_relu_all_func[6] = nullptr;

  int best_config_index = ProfileToGetBestConfig(
      conv2d_bias_add_relu_all_func, params, CONV2D_BIAS_ADD_RELU);
  map_problem_conv2d_bias_add_relu[problem_size] = best_config_index;
  conv2d_bias_add_relu_all_func[best_config_index](params);
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi
