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
#include "paddle/fluid/operators/fused/cudnn_bn_stats_finalize.cu.h"
#include "paddle/fluid/operators/fused/cudnn_scale_bias_add_relu.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/float16.h"

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace op = paddle::operators;
using Tensor = paddle::framework::Tensor;

USE_OP(batch_norm);

// get paddle batchnorm op results as baseline
void BatchNormForwardCompute(const Tensor &x, const Tensor &scale,
                             const Tensor &bias, Tensor *mean, Tensor *var,
                             Tensor *y, Tensor *saved_mean, Tensor *saved_var,
                             Tensor *reserve_space,
                             const framework::DDim &data_dim,
                             const framework::DDim &param_dim,
                             const platform::CUDADeviceContext &ctx) {
  framework::Scope scope;
  auto var_x = scope.Var("X");
  auto tensor_x = var_x->GetMutable<framework::LoDTensor>();
  auto var_scale = scope.Var("Scale");
  auto tensor_scale = var_scale->GetMutable<framework::LoDTensor>();
  auto var_bias = scope.Var("Bias");
  auto tensor_bias = var_bias->GetMutable<framework::LoDTensor>();
  auto var_mean = scope.Var("Mean");
  auto tensor_mean = var_mean->GetMutable<framework::LoDTensor>();
  auto var_var = scope.Var("Variance");
  auto tensor_var = var_var->GetMutable<framework::LoDTensor>();
  auto var_y = scope.Var("Y");
  auto tensor_y = var_y->GetMutable<framework::LoDTensor>();
  auto var_saved_mean = scope.Var("SavedMean");
  auto tensor_saved_mean = var_saved_mean->GetMutable<framework::LoDTensor>();
  auto var_saved_var = scope.Var("SavedVariance");
  auto tensor_saved_var = var_saved_var->GetMutable<framework::LoDTensor>();
  auto var_reserve = scope.Var("ReserveSpace");
  auto tensor_reserve = var_reserve->GetMutable<framework::LoDTensor>();

  auto place = ctx.GetPlace();
  TensorCopySync(x, place, tensor_x);
  TensorCopySync(scale, place, tensor_scale);
  TensorCopySync(bias, place, tensor_bias);
  TensorCopySync(*mean, place, tensor_mean);
  TensorCopySync(*var, place, tensor_var);

  framework::AttributeMap attrs;
  std::string data_layout = "NHWC";
  attrs.insert({"data_layout", data_layout});

  auto op = framework::OpRegistry::CreateOp(
      "batch_norm", {{"X", {"X"}},
                     {"Scale", {"Scale"}},
                     {"Bias", {"Bias"}},
                     {"Mean", {"Mean"}},
                     {"Variance", {"Variance"}}},
      {{"Y", {"Y"}},
       {"MeanOut", {"Mean"}},
       {"VarianceOut", {"Variance"}},
       {"SavedMean", {"SavedMean"}},
       {"SavedVariance", {"SavedVariance"}},
       {"ReserveSpace", {"ReserveSpace"}}},
      attrs);
  op->Run(scope, ctx.GetPlace());

  TensorCopySync(*tensor_y, place, y);
  TensorCopySync(*tensor_mean, place, mean);
  TensorCopySync(*tensor_var, place, var);
  TensorCopySync(*tensor_saved_mean, place, saved_mean);
  TensorCopySync(*tensor_saved_var, place, saved_var);
  TensorCopySync(*tensor_reserve, place, reserve_space);
  ctx.Wait();
}

template <typename T>
class TestCudnnBNAddReluForward {
 public:
  TestCudnnBNAddReluForward() {
    batch_size_ = 2;
    height_ = 8;
    width_ = 8;
    channels_ = 32;
    ele_count_ = batch_size_ * height_ * width_;
    ctx_ = new platform::CUDADeviceContext(place_);
  }

  TestCudnnBNAddReluForward(int batch_size, int height, int width,
                            int channels) {
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    channels_ = channels;
    ele_count_ = batch_size_ * height_ * width_;
    ctx_ = new platform::CUDADeviceContext(place_);
  }

  ~TestCudnnBNAddReluForward() { delete ctx_; }

  void SetUp() {
    data_size_ = batch_size_ * height_ * width_ * channels_;
    param_size_ = channels_;

    x_vec_.resize(data_size_);
    sum_vec_.resize(param_size_);
    sum_of_squares_vec_.resize(param_size_);
    scale_vec_.resize(param_size_);
    bias_vec_.resize(param_size_);
    mean_vec_.resize(param_size_);
    var_vec_.resize(param_size_);
    y_vec_.resize(data_size_);
    saved_mean_vec_.resize(param_size_);
    saved_var_vec_.resize(param_size_);
    equiv_scale_vec_.resize(param_size_);
    equiv_bias_vec_.resize(param_size_);
    base_y_vec_.resize(data_size_);
    base_mean_vec_.resize(param_size_);
    base_var_vec_.resize(param_size_);
    base_saved_mean_vec_.resize(param_size_);
    base_saved_var_vec_.resize(param_size_);

    // initial data
    std::default_random_engine random(0);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int c = 0; c < channels_; ++c) {
      float sum = 0;
      float sum_of_squares = 0;
      for (int n = 0; n < batch_size_; ++n) {
        for (int h = 0; h < height_; ++h) {
          for (int w = 0; w < width_; ++w) {
            float temp = dis(random);
            float ttemp = static_cast<float>(static_cast<T>(temp));
            int idx = n * height_ * width_ * channels_ +
                      h * width_ * channels_ + w * channels_ + c;
            sum += ttemp;
            sum_of_squares += ttemp * ttemp;
            x_vec_[idx] = static_cast<T>(temp);
          }
        }
      }
      sum_vec_[c] = sum;
      sum_of_squares_vec_[c] = sum_of_squares;
    }
    for (int i = 0; i < param_size_; ++i) {
      scale_vec_[i] = 1.0;
      bias_vec_[i] = 0.0;
      mean_vec_[i] = 0.0;
      var_vec_[i] = 1.0;
      saved_mean_vec_[i] = 0.0;
      saved_var_vec_[i] = 0.0;
      base_mean_vec_[i] = 0.0;
      base_var_vec_[i] = 1.0;
      base_saved_mean_vec_[i] = 0.0;
      base_saved_var_vec_[i] = 0.0;
    }
    for (int i = 0; i < data_size_; ++i) {
      y_vec_[i] = static_cast<T>(0.0f);
      base_y_vec_[i] = static_cast<T>(0.0f);
    }

    // input
    framework::TensorFromVector<T>(x_vec_, *ctx_, &x_);
    x_.Resize({batch_size_, height_, width_, channels_});
    framework::TensorFromVector<float>(sum_vec_, *ctx_, &sum_);
    sum_.Resize({1, 1, 1, channels_});
    framework::TensorFromVector<float>(sum_of_squares_vec_, *ctx_,
                                       &sum_of_squares_);
    sum_of_squares_.Resize({1, 1, 1, channels_});
    framework::TensorFromVector<float>(scale_vec_, *ctx_, &scale_);
    scale_.Resize({1, 1, 1, channels_});
    framework::TensorFromVector<float>(bias_vec_, *ctx_, &bias_);
    bias_.Resize({1, 1, 1, channels_});
    framework::TensorFromVector<float>(mean_vec_, *ctx_, &mean_);
    mean_.Resize({1, 1, 1, channels_});
    framework::TensorFromVector<float>(var_vec_, *ctx_, &var_);
    var_.Resize({1, 1, 1, channels_});
    // baseline
    framework::TensorFromVector<float>(scale_vec_, *ctx_, &base_scale_);
    base_scale_.Resize({channels_});
    framework::TensorFromVector<float>(bias_vec_, *ctx_, &base_bias_);
    base_bias_.Resize({channels_});
    framework::TensorFromVector<float>(base_mean_vec_, *ctx_, &base_mean_);
    base_mean_.Resize({channels_});
    framework::TensorFromVector<float>(base_var_vec_, *ctx_, &base_var_);
    base_var_.Resize({channels_});
    // output
    y_.Resize({batch_size_, height_, width_, channels_});
    equiv_scale_.Resize({1, 1, 1, channels_});
    equiv_bias_.Resize({1, 1, 1, channels_});
    saved_mean_.Resize({1, 1, 1, channels_});
    saved_var_.Resize({1, 1, 1, channels_});
    // baseline
    base_y_.Resize({batch_size_, height_, width_, channels_});
    base_saved_mean_.Resize({channels_});
    base_saved_var_.Resize({channels_});
    // bitmask
    int C = channels_;
    int64_t NHW = ele_count_;
    int32_t C_int32Elems = ((C + 63) & ~63) / 32;
    int32_t NHW_int32Elems = (NHW + 31) & ~31;
    bitmask_.Resize({NHW_int32Elems, C_int32Elems, 1});

    ctx_->Wait();
  }

  void BaselineForward() {
    BatchNormForwardCompute(x_, base_scale_, base_bias_, &base_mean_,
                            &base_var_, &base_y_, &base_saved_mean_,
                            &base_saved_var_, &reserve_space_, x_.dims(),
                            framework::make_ddim({channels_}), *ctx_);

    ctx_->Wait();
  }

  // get forward results of cudnn_bn_stats_finalize + cudnn_scale_bias_add_relu
  void FusedForward() {
    auto data_shape = framework::vectorize<int>(x_.dims());
    auto param_shape = framework::vectorize<int>(scale_.dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask_.dims());
    T *x_ptr = x_.data<T>();
    float *sum_ptr = sum_.data<float>();
    float *sum_of_squares_ptr = sum_of_squares_.data<float>();
    float *scale_ptr = scale_.data<float>();
    float *bias_ptr = bias_.data<float>();
    float *mean_ptr = mean_.data<float>();
    float *var_ptr = var_.data<float>();
    float *saved_mean_ptr = saved_mean_.mutable_data<float>(place_);
    float *saved_var_ptr = saved_var_.mutable_data<float>(place_);
    T *equiv_scale_ptr = equiv_scale_.mutable_data<T>(place_);
    T *equiv_bias_ptr = equiv_bias_.mutable_data<T>(place_);
    T *y_ptr = y_.mutable_data<T>(place_);
    int32_t *bitmask_ptr = bitmask_.mutable_data<int32_t>(place_);

    // 1. BN Stats Finalize
    std::shared_ptr<op::CudnnBNStatsFinalizeOp<T>> bn_op(
        new op::CudnnBNStatsFinalizeOp<T>());
    bn_op->Init(*ctx_, param_shape);
    bn_op->Forward(*ctx_, sum_ptr, sum_of_squares_ptr, scale_ptr, bias_ptr,
                   saved_mean_ptr, saved_var_ptr, mean_ptr, var_ptr,
                   equiv_scale_ptr, equiv_bias_ptr, eps_, momentum_, ele_count_,
                   true);
    // 2. Scale Bias + Relu (not fused add)
    std::string act_type = "";
    std::shared_ptr<op::CudnnScaleBiasAddReluOp<T>> sbar_op(
        new op::CudnnScaleBiasAddReluOp<T>(false, false));
    sbar_op->Init(*ctx_, act_type, data_shape, bitmask_shape, data_shape,
                  param_shape);
    sbar_op->Forward(*ctx_, x_ptr, equiv_scale_ptr, equiv_bias_ptr, y_ptr,
                     bitmask_ptr);

    ctx_->Wait();
  }

  void Run() {
    SetUp();
    BaselineForward();
    FusedForward();
  }

  // check forward correctness between baseline and results of fused op.
  void CheckOut(const float diff, bool is_relative_atol = false) {
    TensorToVector(y_, *ctx_, &y_vec_);
    TensorToVector(mean_, *ctx_, &mean_vec_);
    TensorToVector(var_, *ctx_, &var_vec_);
    TensorToVector(saved_mean_, *ctx_, &saved_mean_vec_);
    TensorToVector(saved_var_, *ctx_, &saved_var_vec_);
    TensorToVector(base_y_, *ctx_, &base_y_vec_);
    TensorToVector(base_mean_, *ctx_, &base_mean_vec_);
    TensorToVector(base_var_, *ctx_, &base_var_vec_);
    TensorToVector(base_saved_mean_, *ctx_, &base_saved_mean_vec_);
    TensorToVector(base_saved_var_, *ctx_, &base_saved_var_vec_);
    ctx_->Wait();

    for (int i = 0; i < data_size_; ++i) {
      if (is_relative_atol) {
        EXPECT_LT(std::abs((y_vec_[i] - base_y_vec_[i]) / base_y_vec_[i]),
                  static_cast<T>(diff));
      } else {
        EXPECT_LT(std::abs(y_vec_[i] - base_y_vec_[i]), static_cast<T>(diff));
      }
    }

    for (int i = 0; i < param_size_; ++i) {
      if (is_relative_atol) {
        EXPECT_LT(
            std::abs((mean_vec_[i] - base_mean_vec_[i]) / base_mean_vec_[i]),
            diff);
        EXPECT_LT(std::abs((var_vec_[i] - base_var_vec_[i]) / base_var_vec_[i]),
                  diff);
        EXPECT_LT(std::abs((saved_mean_vec_[i] - base_saved_mean_vec_[i]) /
                           base_saved_mean_vec_[i]),
                  diff);
        EXPECT_LT(std::abs((saved_var_vec_[i] - base_saved_var_vec_[i]) /
                           base_saved_var_vec_[i]),
                  diff);
      } else {
        EXPECT_LT(std::abs(mean_vec_[i] - base_mean_vec_[i]), diff);
        EXPECT_LT(std::abs(var_vec_[i] - base_var_vec_[i]), diff);
        EXPECT_LT(std::abs(saved_mean_vec_[i] - base_saved_mean_vec_[i]), diff);
        EXPECT_LT(std::abs(saved_var_vec_[i] - base_saved_var_vec_[i]), diff);
      }
    }
  }

 private:
  int batch_size_, height_, width_, channels_;
  int data_size_, param_size_;

  framework::Tensor x_, scale_, bias_, mean_, var_, sum_, sum_of_squares_;
  framework::Tensor y_, saved_mean_, saved_var_, equiv_scale_, equiv_bias_,
      bitmask_;
  std::vector<T> x_vec_, y_vec_, equiv_scale_vec_, equiv_bias_vec_;
  std::vector<float> sum_vec_, sum_of_squares_vec_, scale_vec_, bias_vec_;
  std::vector<float> mean_vec_, var_vec_, saved_mean_vec_, saved_var_vec_;
  // baseline
  framework::Tensor base_y_, base_scale_, base_bias_, base_mean_, base_var_,
      base_saved_mean_, base_saved_var_, reserve_space_;
  std::vector<T> base_y_vec_;
  std::vector<float> base_mean_vec_, base_var_vec_, base_saved_mean_vec_,
      base_saved_var_vec_;

  double eps_ = 1e-5;
  float momentum_ = 0.9;
  int ele_count_;
  platform::CUDAPlace place_ = platform::CUDAPlace(0);
  platform::CUDADeviceContext *ctx_ =
      static_cast<platform::CUDADeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
};

TEST(CudnnBNAddReluForward, GPUCudnnBNAddReluForwardFp16) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int channels = 64;
  TestCudnnBNAddReluForward<paddle::platform::float16> test(batch_size, height,
                                                            width, channels);
  test.Run();
  test.CheckOut(2e-3);
}
