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
#include "paddle/fluid/operators/fused/cudnn_norm_conv.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/float16.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace op = paddle::operators;
using Tensor = paddle::framework::Tensor;

USE_OP(conv2d);
USE_OP_DEVICE_KERNEL(conv2d, CUDNN);

template <typename T>
void InitRandomTensor(const std::vector<int64_t> &dims,
                      framework::Tensor *cpu_out) {
  T *cpu_out_ptr = cpu_out->mutable_data<T>(framework::make_ddim(dims),
                                            platform::CPUPlace());

  std::default_random_engine random(0);
  std::uniform_real_distribution<float> dis(0.0, 1.0);
  for (int i = 0; i < cpu_out->numel(); ++i) {
    cpu_out_ptr[i] = static_cast<T>(dis(random));
  }
}

template <typename T>
void TransposeNchwToNhwc(const framework::Tensor &cpu_in,
                         framework::Tensor *cpu_out) {
  auto in_dims = cpu_in.dims();
  EXPECT_EQ(cpu_in.dims().size(), 4);

  const T *cpu_in_ptr = cpu_in.data<T>();
  T *cpu_out_ptr = cpu_out->mutable_data<T>(
      {in_dims[0], in_dims[2], in_dims[3], in_dims[1]}, platform::CPUPlace());

  int64_t n = in_dims[0];
  int64_t c = in_dims[1];
  int64_t hw = in_dims[2] * in_dims[3];
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < hw; ++j) {
      for (int k = 0; k < c; ++k) {
        int dst_idx = i * hw * c + j * c + k;
        int src_idx = i * c * hw + k * hw + j;
        cpu_out_ptr[dst_idx] = cpu_in_ptr[src_idx];
      }
    }
  }
}

template <typename T>
void CheckOutput(const framework::Tensor &cpu_res,
                 const framework::Tensor &cpu_base, T diff,
                 bool is_relative_atol = false) {
  EXPECT_EQ(cpu_res.dims(), cpu_base.dims());

  const T *cpu_res_ptr = cpu_res.data<T>();
  const T *cpu_base_ptr = cpu_base.data<T>();
  for (int i = 0; i < cpu_res.numel(); ++i) {
    if (is_relative_atol) {
      EXPECT_LT(std::abs((cpu_res_ptr[i] - cpu_base_ptr[i]) / cpu_base_ptr[i]),
                diff);
    } else {
      EXPECT_LT(std::abs(cpu_res_ptr[i] - cpu_base_ptr[i]), diff);
    }
  }
}

// get paddle conv2d op results as baseline
template <typename T>
void Conv2DForwardCompute(const platform::CUDADeviceContext &ctx,
                          const Tensor &cpu_x, const Tensor &cpu_w,
                          Tensor *cpu_y) {
  framework::Scope scope;
  auto *x = scope.Var("Input")->GetMutable<framework::LoDTensor>();
  auto *w = scope.Var("Filter")->GetMutable<framework::LoDTensor>();
  auto *y = scope.Var("Output")->GetMutable<framework::LoDTensor>();

  auto place = ctx.GetPlace();
  TensorCopySync(cpu_x, place, x);
  TensorCopySync(cpu_w, place, w);

  framework::AttributeMap attrs;
  bool use_cudnn = true;
  std::string data_format = "NHWC";
  std::string padding_algorithm = "SAME";
  attrs.insert({"use_cudnn", use_cudnn});
  attrs.insert({"data_format", data_format});
  attrs.insert({"padding_algorithm", padding_algorithm});

  auto op = framework::OpRegistry::CreateOp(
      "conv2d", {{"Input", {"Input"}}, {"Filter", {"Filter"}}},
      {{"Output", {"Output"}}}, attrs);
  op->Run(scope, ctx.GetPlace());

  TensorCopySync(*y, platform::CPUPlace(), cpu_y);
}

template <typename T>
class TestCudnnNormConvOpForward {
 public:
  TestCudnnNormConvOpForward(int batch_size, int height, int width,
                             int input_channels, int output_channels,
                             int kernel_size, int stride) {
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    input_channels_ = input_channels;
    output_channels_ = output_channels;
    kernel_size_ = kernel_size;
    stride_ = stride;
    padding_ = (kernel_size_ - 1) / 2;
  }

  ~TestCudnnNormConvOpForward() {}

  void SetUp() {
    InitRandomTensor<T>({batch_size_, height_, width_, input_channels_},
                        &cpu_input_);
    InitRandomTensor<T>(
        {output_channels_, input_channels_, kernel_size_, kernel_size_},
        &cpu_filter_nchw_);
    // transpoes for filter, NCHW -> NHWC
    TransposeNchwToNhwc<T>(cpu_filter_nchw_, &cpu_filter_nhwc_);
  }

  void BaselineForward(const platform::CUDADeviceContext &ctx) {
    Conv2DForwardCompute<T>(ctx, cpu_input_, cpu_filter_nchw_,
                            &cpu_output_base_);
  }

  // get forward results of cudnn_norm_conv
  void FusedForward(const platform::CUDADeviceContext &ctx) {
    framework::Tensor input;
    framework::Tensor filter_nhwc;
    framework::Tensor output;
    framework::Tensor sum;
    framework::Tensor sum_of_squares;

    auto place = ctx.GetPlace();
    TensorCopySync(cpu_input_, place, &input);
    TensorCopySync(cpu_filter_nhwc_, place, &filter_nhwc);

    T *input_ptr = input.data<T>();
    T *filter_ptr = filter_nhwc.data<T>();
    T *output_ptr = output.mutable_data<T>(
        {batch_size_, height_, width_, output_channels_}, place);
    float *sum_ptr =
        sum.mutable_data<float>({1, 1, 1, output_channels_}, place);
    float *sum_of_squares_ptr =
        sum_of_squares.mutable_data<float>({1, 1, 1, output_channels_}, place);

    auto input_shape = framework::vectorize<int>(input.dims());
    auto filter_shape = framework::vectorize<int>(filter_nhwc.dims());
    auto output_shape = framework::vectorize<int>(output.dims());
    op::CudnnNormConvolution<T> conv_op(ctx, input_shape, filter_shape,
                                        output_shape, padding_, stride_,
                                        dilation_, group_);
    conv_op.Forward(ctx, input_ptr, filter_ptr, output_ptr, sum_ptr,
                    sum_of_squares_ptr);

    TensorCopySync(output, platform::CPUPlace(), &cpu_output_);
    TensorCopySync(sum, platform::CPUPlace(), &cpu_sum_);
    TensorCopySync(sum_of_squares, platform::CPUPlace(), &cpu_sum_of_squares_);
  }

  void Run() {
    platform::CUDADeviceContext *ctx =
        static_cast<platform::CUDADeviceContext *>(
            platform::DeviceContextPool::Instance().Get(
                platform::CUDAPlace(0)));
    SetUp();
    BaselineForward(*ctx);
    FusedForward(*ctx);
  }

  // check forward correctness between baseline and results of normconv.
  void CheckOut(const T diff, bool is_relative_atol = false) {
    CheckOutput<T>(cpu_output_, cpu_output_base_, diff, is_relative_atol);
  }

 private:
  int batch_size_;
  int height_;
  int width_;
  int input_channels_;
  int output_channels_;
  int kernel_size_;
  int stride_;
  int padding_;
  const int dilation_ = 1;
  const int group_ = 1;

  framework::Tensor cpu_input_;
  framework::Tensor cpu_filter_nchw_;
  framework::Tensor cpu_filter_nhwc_;
  framework::Tensor cpu_output_;
  framework::Tensor cpu_output_base_;
  framework::Tensor cpu_sum_;
  framework::Tensor cpu_sum_of_squares_;
};

// test for fp16, kernel = 1, output_channels = input_channels
TEST(CudnnNormConvForward, GPUCudnnNormConvForward1Fp16) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 32;
  int kernel_size = 1;
  int stride = 1;
  TestCudnnNormConvOpForward<paddle::platform::float16> test(
      batch_size, height, width, input_channels, output_channels, kernel_size,
      stride);
  test.Run();
  test.CheckOut(static_cast<paddle::platform::float16>(1e-3), true);
}

// test for fp16, kernel = 3, output_channels = input_channels
TEST(CudnnNormConvForward, GPUCudnnNormConvForward2Fp16) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 32;
  int kernel_size = 3;
  int stride = 1;
  TestCudnnNormConvOpForward<paddle::platform::float16> test(
      batch_size, height, width, input_channels, output_channels, kernel_size,
      stride);
  test.Run();
  test.CheckOut(static_cast<paddle::platform::float16>(1e-3), true);
}

// test for fp16, kernel = 1, output_channels = input_channels * 4
TEST(CudnnNormConvForward, GPUCudnnNormConvForward3Fp16) {
  int batch_size = 4;
  int height = 56;
  int width = 56;
  int input_channels = 32;
  int output_channels = 128;
  int kernel_size = 1;
  int stride = 1;
  TestCudnnNormConvOpForward<paddle::platform::float16> test(
      batch_size, height, width, input_channels, output_channels, kernel_size,
      stride);
  test.Run();
  test.CheckOut(static_cast<paddle::platform::float16>(1e-3), true);
}
