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

// get paddle conv2d op results as baseline
template <typename T>
void Conv2DForwardCompute(const Tensor &x, const Tensor &w, Tensor *y,
                          const platform::CUDADeviceContext &ctx) {
  framework::Scope scope;
  auto var_x = scope.Var("Input");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_w = scope.Var("Filter");
  auto tensor_w = var_w->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Output");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();

  auto place = ctx.GetPlace();
  TensorCopySync(x, place, tensor_x);
  TensorCopySync(w, place, tensor_w);

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

  TensorCopySync(*tensor_y, place, y);
  ctx.Wait();
}

template <typename T>
class TestCudnnNormConvOpForward {
 public:
  TestCudnnNormConvOpForward() {
    batch_size_ = 2;
    height_ = 8;
    width_ = 8;
    input_channels_ = 8;
    output_channels_ = 32;
    kernel_size_ = 1;
    stride_ = 1;
    pad_ = 0;
  }

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
    pad_ = (kernel_size_ - 1) / 2;
  }

  ~TestCudnnNormConvOpForward() {}

  void SetUp() {
    input_size_ = batch_size_ * height_ * width_ * input_channels_;
    filter_size_ =
        output_channels_ * input_channels_ * kernel_size_ * kernel_size_;
    output_size_ = batch_size_ * height_ * width_ * output_channels_;
    param_size_ = output_channels_;

    input_vec_.resize(input_size_);
    filter_raw_vec_.resize(filter_size_);
    filter_pro_vec_.resize(filter_size_);

    std::default_random_engine random(0);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < input_size_; ++i) {
      input_vec_[i] = static_cast<T>(dis(random));
    }
    for (int i = 0; i < filter_size_; ++i) {
      filter_raw_vec_[i] = static_cast<T>(dis(random));
    }
    // transpoes for filter
    // NCHW->NHWC
    for (int oc = 0; oc < output_channels_; ++oc) {
      for (int kh = 0; kh < kernel_size_; ++kh) {
        for (int kw = 0; kw < kernel_size_; ++kw) {
          for (int ic = 0; ic < input_channels_; ++ic) {
            int dst_idx = oc * kernel_size_ * kernel_size_ * input_channels_ +
                          kh * kernel_size_ * input_channels_ +
                          kw * input_channels_ + ic;
            int src_idx = oc * kernel_size_ * kernel_size_ * input_channels_ +
                          ic * kernel_size_ * kernel_size_ + kh * kernel_size_ +
                          kw;
            filter_pro_vec_[dst_idx] = filter_raw_vec_[src_idx];
          }
        }
      }
    }

    framework::TensorFromVector<T>(input_vec_, *ctx_, &input_);
    input_.Resize({batch_size_, height_, width_, input_channels_});
    framework::TensorFromVector<T>(filter_raw_vec_, *ctx_, &filter_raw_);
    filter_raw_.Resize(
        {output_channels_, input_channels_, kernel_size_, kernel_size_});
    framework::TensorFromVector<T>(filter_pro_vec_, *ctx_, &filter_pro_);
    filter_pro_.Resize(
        {output_channels_, kernel_size_, kernel_size_, input_channels_});
    output_.Resize({batch_size_, height_, width_, output_channels_});
    base_output_.Resize({batch_size_, height_, width_, output_channels_});
    sum_.Resize({1, 1, 1, output_channels_});
    sum_of_squares_.Resize({1, 1, 1, output_channels_});
    ctx_->Wait();
  }

  void BaselineForward() {
    Conv2DForwardCompute<T>(input_, filter_raw_, &base_output_, *ctx_);
    ctx_->Wait();
  }

  // get forward results of cudnn_norm_conv
  void FusedForward() {
    auto input_shape = framework::vectorize<int>(input_.dims());
    auto filter_shape = framework::vectorize<int>(filter_pro_.dims());
    auto output_shape = framework::vectorize<int>(output_.dims());
    T *input_ptr = input_.data<T>();
    T *filter_ptr = filter_pro_.data<T>();
    T *output_ptr = output_.mutable_data<T>(place_);
    float *sum_ptr = sum_.mutable_data<float>(place_);
    float *sum_of_squares_ptr = sum_of_squares_.mutable_data<float>(place_);

    std::shared_ptr<op::CudnnNormConvolutionOp<T>> conv_op(
        new op::CudnnNormConvolutionOp<T>());
    conv_op->Init(*ctx_, input_shape, filter_shape, output_shape, pad_, stride_,
                  dilate_, group_);
    conv_op->Forward(*ctx_, input_ptr, filter_ptr, output_ptr, sum_ptr,
                     sum_of_squares_ptr);
    ctx_->Wait();
  }

  void Run() {
    SetUp();
    BaselineForward();
    FusedForward();
  }

  // check forward correctness between baseline and results of normconv.
  void CheckOut(const T diff, bool is_relative_atol = false) {
    std::vector<T> base_output_vec, output_vec;
    output_vec.resize(output_size_);
    base_output_vec.resize(output_size_);
    TensorToVector(base_output_, *ctx_, &base_output_vec);
    TensorToVector(output_, *ctx_, &output_vec);
    ctx_->Wait();

    for (int i = 0; i < output_size_; ++i) {
      if (is_relative_atol) {
        EXPECT_LT(
            std::abs((output_vec[i] - base_output_vec[i]) / base_output_vec[i]),
            diff);
      } else {
        EXPECT_LT(std::abs(output_vec[i] - base_output_vec[i]), diff);
      }
    }
  }

 private:
  int batch_size_, height_, width_, input_channels_, output_channels_;
  int kernel_size_, stride_, pad_;
  const int dilate_ = 1;
  const int group_ = 1;
  int input_size_, filter_size_, output_size_, param_size_;

  framework::Tensor input_, filter_raw_, filter_pro_, output_, base_output_;
  framework::Tensor sum_, sum_of_squares_;
  std::vector<T> input_vec_, filter_raw_vec_, filter_pro_vec_;

  platform::CUDAPlace place_ = platform::CUDAPlace(0);
  platform::CUDADeviceContext *ctx_ =
      static_cast<platform::CUDADeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
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
