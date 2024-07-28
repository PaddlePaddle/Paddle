/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/fusion/gpu/cudnn_norm_conv.cu.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;

USE_OP_ITSELF(conv2d);
USE_OP_ITSELF(conv2d_grad);
PD_DECLARE_KERNEL(conv2d, GPUDNN, ALL_LAYOUT);
PD_DECLARE_KERNEL(conv2d_grad, GPUDNN, ALL_LAYOUT);

template <typename T>
void InitRandomTensor(const std::vector<int64_t> &dims,
                      phi::DenseTensor *cpu_out) {
  T *cpu_out_ptr =
      cpu_out->mutable_data<T>(common::make_ddim(dims), phi::CPUPlace());

  std::default_random_engine random(0);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int i = 0; i < cpu_out->numel(); ++i) {
    cpu_out_ptr[i] = static_cast<T>(dis(random));
  }
}

template <typename T>
void TransposeNchwToNhwc(const phi::DenseTensor &cpu_in,
                         phi::DenseTensor *cpu_out) {
  const auto &in_dims = cpu_in.dims();
  EXPECT_EQ(cpu_in.dims().size(), 4);

  const T *cpu_in_ptr = cpu_in.data<T>();
  T *cpu_out_ptr = cpu_out->mutable_data<T>(
      {in_dims[0], in_dims[2], in_dims[3], in_dims[1]}, phi::CPUPlace());

  int64_t n = in_dims[0];
  int64_t c = in_dims[1];
  int64_t hw = in_dims[2] * in_dims[3];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < hw; ++j) {
      for (int k = 0; k < c; ++k) {
        int dst_idx = i * hw * c + j * c + k;   // NOLINT
        int src_idx = i * c * hw + k * hw + j;  // NOLINT
        cpu_out_ptr[dst_idx] = cpu_in_ptr[src_idx];
      }
    }
  }
}

template <typename T>
void CheckOutput(const phi::DenseTensor &cpu_res,
                 const phi::DenseTensor &cpu_base,
                 float diff,
                 bool is_relative_atol = false) {
  EXPECT_EQ(cpu_res.dims(), cpu_base.dims());

  const T *cpu_res_ptr = cpu_res.data<T>();
  const T *cpu_base_ptr = cpu_base.data<T>();
  for (int i = 0; i < cpu_res.numel(); ++i) {
    if (is_relative_atol) {
      EXPECT_LT(static_cast<float>(std::abs((cpu_res_ptr[i] - cpu_base_ptr[i]) /
                                            cpu_base_ptr[i])),
                diff);
    } else {
      EXPECT_LT(static_cast<float>(std::abs(cpu_res_ptr[i] - cpu_base_ptr[i])),
                diff);
    }
  }
}

// Use Paddle conv2d op results as baseline
void ComputeConv2DForward(const phi::GPUContext &ctx,
                          const phi::DenseTensor &cpu_input,
                          const phi::DenseTensor &cpu_filter,
                          phi::DenseTensor *cpu_output,
                          int stride,
                          int padding) {
  framework::Scope scope;
  auto *input = scope.Var("Input")->GetMutable<phi::DenseTensor>();
  auto *filter = scope.Var("Filter")->GetMutable<phi::DenseTensor>();
  auto *output = scope.Var("Output")->GetMutable<phi::DenseTensor>();

  auto place = ctx.GetPlace();
  paddle::framework::TensorCopySync(cpu_input, place, input);
  paddle::framework::TensorCopySync(cpu_filter, place, filter);

  framework::AttributeMap attrs;
  bool use_cudnn = true;
  std::string data_format = "NHWC";
  std::vector<int> strides = {stride, stride};
  std::vector<int> paddings = {padding, padding};
  attrs.insert({"strides", strides});
  attrs.insert({"paddings", paddings});
  attrs.insert({"use_cudnn", use_cudnn});
  attrs.insert({"data_format", data_format});

  auto op = framework::OpRegistry::CreateOp(
      "conv2d",
      {{"Input", {"Input"}}, {"Filter", {"Filter"}}},
      {{"Output", {"Output"}}},
      attrs);
  op->Run(scope, ctx.GetPlace());

  paddle::framework::TensorCopySync(*output, phi::CPUPlace(), cpu_output);
}

// Use Paddle conv2d_grad op results as baseline
void ComputeConv2DBackward(const phi::GPUContext &ctx,
                           const phi::DenseTensor &cpu_input,
                           const phi::DenseTensor &cpu_filter,
                           const phi::DenseTensor &cpu_output_grad,
                           phi::DenseTensor *cpu_input_grad,
                           phi::DenseTensor *cpu_filter_grad,
                           int stride,
                           int padding,
                           int dilation) {
  framework::Scope scope;
  auto *input = scope.Var("Input")->GetMutable<phi::DenseTensor>();
  auto *filter = scope.Var("Filter")->GetMutable<phi::DenseTensor>();
  auto *output_grad = scope.Var("Output@GRAD")->GetMutable<phi::DenseTensor>();
  auto *input_grad = scope.Var("Input@GRAD")->GetMutable<phi::DenseTensor>();
  auto *filter_grad = scope.Var("Filter@GRAD")->GetMutable<phi::DenseTensor>();

  auto place = ctx.GetPlace();
  paddle::framework::TensorCopySync(cpu_input, place, input);
  paddle::framework::TensorCopySync(cpu_filter, place, filter);
  paddle::framework::TensorCopySync(cpu_output_grad, place, output_grad);

  framework::AttributeMap attrs;
  bool use_cudnn = true;
  std::string data_format = "NHWC";
  std::string padding_algorithm = "EXPLICIT";
  std::vector<int> strides = {stride, stride};
  std::vector<int> paddings = {padding, padding};
  std::vector<int> dilations = {dilation, dilation};
  int groups = 1;
  bool exhaustive_search = false;
  bool use_addto = false;
  attrs.insert({"use_cudnn", use_cudnn});
  attrs.insert({"data_format", data_format});
  attrs.insert({"padding_algorithm", padding_algorithm});
  attrs.insert({"strides", strides});
  attrs.insert({"paddings", paddings});
  attrs.insert({"dilations", dilations});
  attrs.insert({"groups", groups});
  attrs.insert({"exhaustive_search", exhaustive_search});
  attrs.insert({"use_addto", use_addto});
  attrs.insert({"workspace_size_MB", 512});

  auto op = framework::OpRegistry::CreateOp(
      "conv2d_grad",
      {{"Input", {"Input"}},
       {"Filter", {"Filter"}},
       {"Output@GRAD", {"Output@GRAD"}}},
      {{"Input@GRAD", {"Input@GRAD"}}, {"Filter@GRAD", {"Filter@GRAD"}}},
      attrs);
  op->Run(scope, ctx.GetPlace());

  paddle::framework::TensorCopySync(
      *input_grad, phi::CPUPlace(), cpu_input_grad);
  paddle::framework::TensorCopySync(
      *filter_grad, phi::CPUPlace(), cpu_filter_grad);
}

template <typename T>
void ComputeSumAndSquareSum(const phi::DenseTensor &cpu_out,
                            phi::DenseTensor *cpu_sum,
                            phi::DenseTensor *cpu_sum_of_square) {
  const auto &dims = cpu_out.dims();
  int64_t c = dims[3];

  const T *cpu_out_ptr = cpu_out.data<T>();
  float *cpu_sum_ptr =
      cpu_sum->mutable_data<float>({1, 1, 1, c}, phi::CPUPlace());
  float *cpu_sum_square_ptr =
      cpu_sum_of_square->mutable_data<float>({1, 1, 1, c}, phi::CPUPlace());

  for (int j = 0; j < c; ++j) {
    float tmp_sum = 0.0f;
    float tmp_sum_of_squares = 0.0f;
    for (int i = 0; i < cpu_out.numel() / c; ++i) {
      float tmp_out = static_cast<float>(cpu_out_ptr[i * c + j]);
      tmp_sum += tmp_out;
      tmp_sum_of_squares += tmp_out * tmp_out;
    }
    cpu_sum_ptr[j] = tmp_sum;
    cpu_sum_square_ptr[j] = tmp_sum_of_squares;
  }
}

template <typename T>
class CudnnNormConvolutionTester {
 public:
  CudnnNormConvolutionTester(int batch_size,
                             int height,
                             int width,
                             int input_channels,
                             int output_channels,
                             int kernel_size,
                             int stride) {
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    input_channels_ = input_channels;
    output_channels_ = output_channels;
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = (kernel_size_ - 1) / 2;
    out_height_ = (height_ + 2 * padding_ - kernel_size_) / stride_ + 1;
    out_width_ = (width_ + 2 * padding_ - kernel_size_) / stride_ + 1;
    SetUp();
  }

  ~CudnnNormConvolutionTester() = default;

  void CheckForward(float diff, bool is_relative_atol = false) {
    phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

    phi::DenseTensor cpu_output_base;
    phi::DenseTensor cpu_sum_base;
    phi::DenseTensor cpu_sum_of_square_base;
    BaselineForward(
        *ctx, &cpu_output_base, &cpu_sum_base, &cpu_sum_of_square_base);

    phi::DenseTensor cpu_output;
    phi::DenseTensor cpu_sum;
    phi::DenseTensor cpu_sum_of_square;
    FusedForward(*ctx, &cpu_output, &cpu_sum, &cpu_sum_of_square);

    // Check forward correctness between baseline and results of normconv.
    CheckOutput<T>(cpu_output, cpu_output_base, diff, is_relative_atol);
    CheckOutput<float>(cpu_sum, cpu_sum_base, diff, is_relative_atol);
    CheckOutput<float>(
        cpu_sum_of_square, cpu_sum_of_square_base, diff, is_relative_atol);
  }

  void CheckBackward(float diff, bool is_relative_atol = false) {
    phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
        phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

    phi::DenseTensor cpu_input_grad_base;
    phi::DenseTensor cpu_filter_nchw_grad_base;
    phi::DenseTensor cpu_filter_nhwc_grad_base;
    BaselineBackward(*ctx, &cpu_input_grad_base, &cpu_filter_nchw_grad_base);
    TransposeNchwToNhwc<T>(cpu_filter_nchw_grad_base,
                           &cpu_filter_nhwc_grad_base);

    phi::DenseTensor cpu_input_grad;
    phi::DenseTensor cpu_filter_nhwc_grad;
    FusedBackward(*ctx, &cpu_input_grad, &cpu_filter_nhwc_grad);

    // Check backward correctness between baseline and results of normconv.
    CheckOutput<T>(cpu_input_grad, cpu_input_grad_base, diff, is_relative_atol);
    CheckOutput<T>(cpu_filter_nhwc_grad,
                   cpu_filter_nhwc_grad_base,
                   diff,
                   is_relative_atol);
  }

 private:
  void SetUp() {
    InitRandomTensor<T>({batch_size_, height_, width_, input_channels_},
                        &cpu_input_);
    InitRandomTensor<T>(
        {output_channels_, input_channels_, kernel_size_, kernel_size_},
        &cpu_filter_nchw_);
    // transpoes for filter, NCHW -> NHWC
    TransposeNchwToNhwc<T>(cpu_filter_nchw_, &cpu_filter_nhwc_);
    InitRandomTensor<T>(
        {batch_size_, out_height_, out_width_, output_channels_},
        &cpu_output_grad_);
  }

  void BaselineForward(const phi::GPUContext &ctx,
                       phi::DenseTensor *cpu_output_base,
                       phi::DenseTensor *cpu_sum_base,
                       phi::DenseTensor *cpu_sum_of_square_base) {
    ComputeConv2DForward(
        ctx, cpu_input_, cpu_filter_nchw_, cpu_output_base, stride_, padding_);
    ComputeSumAndSquareSum<T>(
        *cpu_output_base, cpu_sum_base, cpu_sum_of_square_base);
  }

  void BaselineBackward(const phi::GPUContext &ctx,
                        phi::DenseTensor *cpu_input_grad_base,
                        phi::DenseTensor *cpu_filter_grad_base) {
    ComputeConv2DBackward(ctx,
                          cpu_input_,
                          cpu_filter_nchw_,
                          cpu_output_grad_,
                          cpu_input_grad_base,
                          cpu_filter_grad_base,
                          stride_,
                          padding_,
                          dilation_);
  }

  // get forward results of cudnn_norm_conv
  void FusedForward(const phi::GPUContext &ctx,
                    phi::DenseTensor *cpu_output,
                    phi::DenseTensor *cpu_sum,
                    phi::DenseTensor *cpu_sum_of_square) {
    phi::DenseTensor input;
    phi::DenseTensor filter_nhwc;
    phi::DenseTensor output;
    phi::DenseTensor sum;
    phi::DenseTensor sum_of_square;

    auto place = ctx.GetPlace();
    paddle::framework::TensorCopySync(cpu_input_, place, &input);
    paddle::framework::TensorCopySync(cpu_filter_nhwc_, place, &filter_nhwc);

    output.Resize(common::make_ddim(
        {batch_size_, out_height_, out_width_, output_channels_}));
    sum.Resize(common::make_ddim({1, 1, 1, output_channels_}));
    sum_of_square.Resize(common::make_ddim({1, 1, 1, output_channels_}));

    auto input_shape = common::vectorize<int>(input.dims());
    auto filter_shape = common::vectorize<int>(filter_nhwc.dims());
    auto output_shape = common::vectorize<int>(output.dims());
    phi::fusion::CudnnNormConvolution<T> conv_op(ctx,
                                                 input_shape,
                                                 filter_shape,
                                                 output_shape,
                                                 padding_,
                                                 stride_,
                                                 dilation_,
                                                 group_);
    conv_op.Forward(ctx, input, filter_nhwc, &output, &sum, &sum_of_square);

    paddle::framework::TensorCopySync(output, phi::CPUPlace(), cpu_output);
    paddle::framework::TensorCopySync(sum, phi::CPUPlace(), cpu_sum);
    paddle::framework::TensorCopySync(
        sum_of_square, phi::CPUPlace(), cpu_sum_of_square);
  }

  void FusedBackward(const phi::GPUContext &ctx,
                     phi::DenseTensor *cpu_input_grad,
                     phi::DenseTensor *cpu_filter_grad) {
    phi::DenseTensor input;
    phi::DenseTensor filter_nhwc;
    phi::DenseTensor output_grad;
    phi::DenseTensor input_grad;
    phi::DenseTensor filter_grad;

    auto place = ctx.GetPlace();
    paddle::framework::TensorCopySync(cpu_input_, place, &input);
    paddle::framework::TensorCopySync(cpu_filter_nhwc_, place, &filter_nhwc);
    paddle::framework::TensorCopySync(cpu_output_grad_, place, &output_grad);

    input_grad.Resize(input.dims());
    filter_grad.Resize(filter_nhwc.dims());

    auto input_shape = common::vectorize<int>(input.dims());
    auto filter_shape = common::vectorize<int>(filter_nhwc.dims());
    auto output_shape = common::vectorize<int>(output_grad.dims());
    phi::fusion::CudnnNormConvolutionGrad<T> conv_grad_op(ctx,
                                                          input_shape,
                                                          filter_shape,
                                                          output_shape,
                                                          padding_,
                                                          stride_,
                                                          dilation_,
                                                          group_);
    conv_grad_op.Backward(
        ctx, input, filter_nhwc, output_grad, &input_grad, &filter_grad);

    paddle::framework::TensorCopySync(
        input_grad, phi::CPUPlace(), cpu_input_grad);
    paddle::framework::TensorCopySync(
        filter_grad, phi::CPUPlace(), cpu_filter_grad);
  }

 private:
  int batch_size_;
  int height_;
  int width_;
  int out_height_;
  int out_width_;
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  int stride_;
  int padding_;
  const int dilation_ = 1;
  const int group_ = 1;

  // Forward input
  phi::DenseTensor cpu_input_;
  phi::DenseTensor cpu_filter_nchw_;
  phi::DenseTensor cpu_filter_nhwc_;

  // Backward input
  phi::DenseTensor cpu_output_grad_;
};

// test for fp16, kernel = 1, output_channels = input_channels
TEST(CudnnNormConvFp16, K1S1) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 32;
  int kernel_size = 1;
  int stride = 1;
  CudnnNormConvolutionTester<phi::dtype::float16> test(batch_size,
                                                       height,
                                                       width,
                                                       input_channels,
                                                       output_channels,
                                                       kernel_size,
                                                       stride);
  phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

  if (ctx->GetComputeCapability() < 70 || ctx->GetComputeCapability() >= 90) {
    ASSERT_THROW(test.CheckForward(1e-3, true),
                 paddle::platform::EnforceNotMet);
    ASSERT_THROW(test.CheckBackward(1e-3, true),
                 paddle::platform::EnforceNotMet);
  } else {
    ASSERT_NO_THROW(test.CheckForward(1e-3, true));
    ASSERT_NO_THROW(test.CheckBackward(1e-3, true));
  }
}

// test for fp16, kernel = 3, output_channels = input_channels
TEST(CudnnNormConvFp16, K3S1) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 32;
  int kernel_size = 3;
  int stride = 1;
  CudnnNormConvolutionTester<phi::dtype::float16> test(batch_size,
                                                       height,
                                                       width,
                                                       input_channels,
                                                       output_channels,
                                                       kernel_size,
                                                       stride);
  phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

  if (ctx->GetComputeCapability() < 70 || ctx->GetComputeCapability() >= 90) {
    ASSERT_THROW(test.CheckForward(1e-3, true),
                 paddle::platform::EnforceNotMet);
    ASSERT_THROW(test.CheckBackward(1e-3, true),
                 paddle::platform::EnforceNotMet);
  } else {
    ASSERT_NO_THROW(test.CheckForward(1e-3, true));
    ASSERT_NO_THROW(test.CheckBackward(1e-3, true));
  }
}

// test for fp16, kernel = 1, output_channels = input_channels * 4
TEST(CudnnNormConvFp16, K1S1O4) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 128;
  int kernel_size = 1;
  int stride = 1;
  CudnnNormConvolutionTester<phi::dtype::float16> test(batch_size,
                                                       height,
                                                       width,
                                                       input_channels,
                                                       output_channels,
                                                       kernel_size,
                                                       stride);
  phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

  if (ctx->GetComputeCapability() < 70 || ctx->GetComputeCapability() >= 90) {
    ASSERT_THROW(test.CheckForward(1e-3, true),
                 paddle::platform::EnforceNotMet);
    ASSERT_THROW(test.CheckBackward(1e-3, true),
                 paddle::platform::EnforceNotMet);
  } else {
    ASSERT_NO_THROW(test.CheckForward(1e-3, true));
    ASSERT_NO_THROW(test.CheckBackward(1e-3, true));
  }
}

// test for fp16, kernel = 1, stride = 2, output_channels = input_channels * 4
TEST(CudnnNormConvFp16, K1S2O4) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int input_channels = 32;
  int output_channels = 128;
  int kernel_size = 1;
  int stride = 2;
  CudnnNormConvolutionTester<phi::dtype::float16> test(batch_size,
                                                       height,
                                                       width,
                                                       input_channels,
                                                       output_channels,
                                                       kernel_size,
                                                       stride);
  phi::GPUContext *ctx = static_cast<phi::GPUContext *>(
      phi::DeviceContextPool::Instance().Get(phi::GPUPlace(0)));

  if (ctx->GetComputeCapability() <= 70 || ctx->GetComputeCapability() >= 90) {
    ASSERT_THROW(test.CheckForward(1e-3, true),
                 paddle::platform::EnforceNotMet);
    ASSERT_THROW(test.CheckBackward(1e-3), paddle::platform::EnforceNotMet);
  } else {
    ASSERT_NO_THROW(test.CheckForward(1e-3, true));
    ASSERT_NO_THROW(test.CheckBackward(1e-3));
  }
}
