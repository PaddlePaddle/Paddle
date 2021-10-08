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

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace op = paddle::operators;
using Tensor = paddle::framework::Tensor;

USE_OP(batch_norm);

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
void InitConstantTensor(const std::vector<int64_t> &dims, T value,
                        framework::Tensor *cpu_out) {
  T *cpu_out_ptr = cpu_out->mutable_data<T>(framework::make_ddim(dims),
                                            platform::CPUPlace());
  for (int i = 0; i < cpu_out->numel(); ++i) {
    cpu_out_ptr[i] = value;
  }
}

template <typename T>
void CheckOutput(std::string name, const framework::Tensor &cpu_res,
                 const framework::Tensor &cpu_base, float diff,
                 bool is_relative_atol = false) {
  if (cpu_res.dims().size() == cpu_base.dims().size()) {
    EXPECT_EQ(cpu_res.dims(), cpu_base.dims());
  } else {
    EXPECT_EQ(cpu_res.numel(), cpu_base.numel());
  }

  const T *cpu_res_ptr = cpu_res.data<T>();
  const T *cpu_base_ptr = cpu_base.data<T>();
  float max_diff = 0;
  int index = 0;
  for (int i = 0; i < cpu_res.numel(); ++i) {
    float cur_diff;
    if (is_relative_atol) {
      cur_diff = static_cast<float>(
          std::abs((cpu_res_ptr[i] - cpu_base_ptr[i]) / cpu_base_ptr[i]));
      EXPECT_LT(static_cast<float>(std::abs((cpu_res_ptr[i] - cpu_base_ptr[i]) /
                                            cpu_base_ptr[i])),
                diff);
    } else {
      cur_diff = static_cast<float>(std::abs(cpu_res_ptr[i] - cpu_base_ptr[i]));
      EXPECT_LT(static_cast<float>(std::abs(cpu_res_ptr[i] - cpu_base_ptr[i])),
                diff);
    }
    if (cur_diff > max_diff) {
      max_diff = cur_diff;
      index = i;
    }
  }
  std::string error_type = is_relative_atol ? "relative" : "absolute";
  LOG(INFO) << "[" << name << "], The dims is [" << cpu_res.dims()
            << "], maximum " << error_type << " error is " << max_diff << ": "
            << cpu_res_ptr[index] << " vs " << cpu_base_ptr[index];
}

template <typename T>
void ComputeSumAndSquareSum(const framework::Tensor &cpu_x,
                            framework::Tensor *cpu_sum,
                            framework::Tensor *cpu_sum_of_square) {
  // x is in NHWC format.
  auto dims = cpu_x.dims();
  int64_t c = dims[3];

  const T *cpu_x_ptr = cpu_x.data<T>();
  float *cpu_sum_ptr =
      cpu_sum->mutable_data<float>({1, 1, 1, c}, platform::CPUPlace());
  float *cpu_sum_square_ptr = cpu_sum_of_square->mutable_data<float>(
      {1, 1, 1, c}, platform::CPUPlace());

  for (int j = 0; j < c; ++j) {
    float tmp_sum = 0.0f;
    float tmp_sum_of_squares = 0.0f;
    for (int i = 0; i < cpu_x.numel() / c; ++i) {
      float tmp_x = static_cast<float>(cpu_x_ptr[i * c + j]);
      tmp_sum += tmp_x;
      tmp_sum_of_squares += tmp_x * tmp_x;
    }
    cpu_sum_ptr[j] = tmp_sum;
    cpu_sum_square_ptr[j] = tmp_sum_of_squares;
  }
}

// get paddle batchnorm op results as baseline
void ComputeBatchNormForward(const platform::CUDADeviceContext &ctx,
                             const Tensor &cpu_x, const Tensor &cpu_scale,
                             const Tensor &cpu_bias, Tensor *cpu_mean,
                             Tensor *cpu_var, Tensor *cpu_saved_mean,
                             Tensor *cpu_saved_var, Tensor *cpu_y,
                             Tensor *cpu_reserve_space) {
  framework::Scope scope;
  auto *x = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto *scale = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  auto *bias = scope.Var("Bias")->GetMutable<framework::LoDTensor>();
  auto *mean = scope.Var("Mean")->GetMutable<framework::LoDTensor>();
  auto *var = scope.Var("Variance")->GetMutable<framework::LoDTensor>();
  auto *y = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto *saved_mean = scope.Var("SavedMean")->GetMutable<framework::LoDTensor>();
  auto *saved_var =
      scope.Var("SavedVariance")->GetMutable<framework::LoDTensor>();
  auto *reserve_space =
      scope.Var("ReserveSpace")->GetMutable<framework::LoDTensor>();

  auto place = ctx.GetPlace();
  TensorCopySync(cpu_x, place, x);
  TensorCopySync(cpu_scale, place, scale);
  TensorCopySync(cpu_bias, place, bias);
  TensorCopySync(*cpu_mean, place, mean);
  TensorCopySync(*cpu_var, place, var);

  int64_t channels = x->dims()[3];
  scale->Resize({channels});
  bias->Resize({channels});
  mean->Resize({channels});
  var->Resize({channels});

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

  TensorCopySync(*y, platform::CPUPlace(), cpu_y);
  TensorCopySync(*mean, platform::CPUPlace(), cpu_mean);
  TensorCopySync(*var, platform::CPUPlace(), cpu_var);
  TensorCopySync(*saved_mean, platform::CPUPlace(), cpu_saved_mean);
  TensorCopySync(*saved_var, platform::CPUPlace(), cpu_saved_var);
  TensorCopySync(*reserve_space, platform::CPUPlace(), cpu_reserve_space);
}

template <typename T>
class CudnnBNAddReluTester {
 public:
  CudnnBNAddReluTester(int batch_size, int height, int width, int channels) {
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    channels_ = channels;
    ele_count_ = batch_size_ * height_ * width_;
    SetUp();
  }

  ~CudnnBNAddReluTester() {}

  void CheckForward(float diff, bool is_relative_atol = false) {
    platform::CUDADeviceContext *ctx =
        static_cast<platform::CUDADeviceContext *>(
            platform::DeviceContextPool::Instance().Get(
                platform::CUDAPlace(0)));

    framework::Tensor cpu_mean_base;
    framework::Tensor cpu_var_base;
    framework::Tensor cpu_saved_mean_base;
    framework::Tensor cpu_saved_var_base;
    framework::Tensor cpu_y_base;
    framework::Tensor cpu_reserve_space_base;
    BaselineForward(*ctx, &cpu_mean_base, &cpu_var_base, &cpu_saved_mean_base,
                    &cpu_saved_var_base, &cpu_y_base, &cpu_reserve_space_base);

    framework::Tensor cpu_mean;
    framework::Tensor cpu_var;
    framework::Tensor cpu_saved_mean;
    framework::Tensor cpu_saved_var;
    framework::Tensor cpu_y;
    framework::Tensor cpu_bitmask;
    FusedForward(*ctx, &cpu_mean, &cpu_var, &cpu_saved_mean, &cpu_saved_var,
                 &cpu_y, &cpu_bitmask);

    CheckOutput<float>("Mean", cpu_mean, cpu_mean_base, diff, is_relative_atol);
    CheckOutput<float>("Variance", cpu_var, cpu_var_base, diff,
                       is_relative_atol);
    CheckOutput<float>("SavedMean", cpu_saved_mean, cpu_saved_mean_base, diff,
                       is_relative_atol);
    CheckOutput<float>("SavedVariance", cpu_saved_var, cpu_saved_var_base, diff,
                       is_relative_atol);
    CheckOutput<T>("Y", cpu_y, cpu_y_base, diff, is_relative_atol);
  }

 private:
  void SetUp() {
    // Initialize input data
    InitRandomTensor<T>({batch_size_, height_, width_, channels_}, &cpu_x_);
    ComputeSumAndSquareSum<T>(cpu_x_, &cpu_sum_, &cpu_sum_of_square_);

    // scale and bias should be initialized randomly.
    InitConstantTensor<float>({channels_}, static_cast<float>(1.0f),
                              &cpu_bn_scale_);
    InitConstantTensor<float>({channels_}, static_cast<float>(0.0f),
                              &cpu_bn_bias_);
  }

  void InitMeanVar(Tensor *cpu_mean, Tensor *cpu_var, Tensor *cpu_saved_mean,
                   Tensor *cpu_saved_var) {
    InitConstantTensor<float>({channels_}, static_cast<float>(0.0f), cpu_mean);
    InitConstantTensor<float>({channels_}, static_cast<float>(1.0f), cpu_var);
    InitConstantTensor<float>({channels_}, static_cast<float>(0.0f),
                              cpu_saved_mean);
    InitConstantTensor<float>({channels_}, static_cast<float>(0.0f),
                              cpu_saved_var);
  }

  void BaselineForward(const platform::CUDADeviceContext &ctx, Tensor *cpu_mean,
                       Tensor *cpu_var, Tensor *cpu_saved_mean,
                       Tensor *cpu_saved_var, Tensor *cpu_y,
                       Tensor *cpu_reserve_space) {
    InitMeanVar(cpu_mean, cpu_var, cpu_saved_mean, cpu_saved_var);
    ComputeBatchNormForward(ctx, cpu_x_, cpu_bn_scale_, cpu_bn_bias_, cpu_mean,
                            cpu_var, cpu_saved_mean, cpu_saved_var, cpu_y,
                            cpu_reserve_space);
  }

  // Get forward results of CudnnBNStatsFinalize + CudnnScaleBiasAddRelu
  void FusedForward(const platform::CUDADeviceContext &ctx, Tensor *cpu_mean,
                    Tensor *cpu_var, Tensor *cpu_saved_mean,
                    Tensor *cpu_saved_var, Tensor *cpu_y, Tensor *cpu_bitmask) {
    framework::Tensor x;
    framework::Tensor sum;
    framework::Tensor sum_of_square;
    framework::Tensor bn_scale;
    framework::Tensor bn_bias;

    auto place = ctx.GetPlace();
    TensorCopySync(cpu_x_, place, &x);
    TensorCopySync(cpu_sum_, place, &sum);
    TensorCopySync(cpu_sum_of_square_, place, &sum_of_square);
    TensorCopySync(cpu_bn_scale_, place, &bn_scale);
    TensorCopySync(cpu_bn_bias_, place, &bn_bias);

    bn_scale.Resize({1, 1, 1, channels_});
    bn_bias.Resize({1, 1, 1, channels_});

    T *x_ptr = x.data<T>();
    float *sum_ptr = sum.data<float>();
    float *sum_of_square_ptr = sum_of_square.data<float>();
    float *bn_scale_ptr = bn_scale.data<float>();
    float *bn_bias_ptr = bn_bias.data<float>();

    framework::Tensor mean;
    framework::Tensor var;
    framework::Tensor saved_mean;
    framework::Tensor saved_var;
    framework::Tensor equiv_scale;
    framework::Tensor equiv_bias;
    framework::Tensor y;
    framework::Tensor bitmask;

    InitMeanVar(cpu_mean, cpu_var, cpu_saved_mean, cpu_saved_var);
    TensorCopySync(*cpu_mean, place, &mean);
    TensorCopySync(*cpu_var, place, &var);

    mean.Resize({1, 1, 1, channels_});
    var.Resize({1, 1, 1, channels_});

    float *mean_ptr = mean.data<float>();
    float *var_ptr = var.data<float>();
    float *saved_mean_ptr =
        saved_mean.mutable_data<float>({1, 1, 1, channels_}, place);
    float *saved_var_ptr =
        saved_var.mutable_data<float>({1, 1, 1, channels_}, place);
    T *equiv_scale_ptr =
        equiv_scale.mutable_data<T>({1, 1, 1, channels_}, place);
    T *equiv_bias_ptr = equiv_bias.mutable_data<T>({1, 1, 1, channels_}, place);
    T *y_ptr =
        y.mutable_data<T>({batch_size_, height_, width_, channels_}, place);

    // bitmask
    int c = channels_;
    int64_t nhw = ele_count_;
    int32_t c_int32_elems = ((c + 63) & ~63) / 32;
    int32_t nhw_int32_elems = (nhw + 31) & ~31;
    int32_t *bitmask_ptr = bitmask.mutable_data<int32_t>(
        {nhw_int32_elems, c_int32_elems, 1}, place);

    auto data_shape = framework::vectorize<int>(x.dims());
    auto param_shape = framework::vectorize<int>(bn_scale.dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask.dims());

    // 1. BN Stats Finalize
    op::CudnnBNStatsFinalizeOp<T> bn_op;
    bn_op.Init(ctx, param_shape);
    bn_op.Forward(ctx, sum_ptr, sum_of_square_ptr, bn_scale_ptr, bn_bias_ptr,
                  saved_mean_ptr, saved_var_ptr, mean_ptr, var_ptr,
                  equiv_scale_ptr, equiv_bias_ptr, eps_, momentum_, ele_count_,
                  true);

    // 2. Scale Bias + Relu (not fused add)
    std::string act_type = "";
    op::CudnnScaleBiasAddReluOp<T> sbar_op(false, false);
    sbar_op.Init(ctx, act_type, data_shape, bitmask_shape, data_shape,
                 param_shape);
    sbar_op.Forward(ctx, x_ptr, equiv_scale_ptr, equiv_bias_ptr, y_ptr,
                    bitmask_ptr);

    TensorCopySync(mean, platform::CPUPlace(), cpu_mean);
    TensorCopySync(var, platform::CPUPlace(), cpu_var);
    TensorCopySync(saved_mean, platform::CPUPlace(), cpu_saved_mean);
    TensorCopySync(saved_var, platform::CPUPlace(), cpu_saved_var);
    TensorCopySync(y, platform::CPUPlace(), cpu_y);
    TensorCopySync(bitmask, platform::CPUPlace(), cpu_bitmask);
  }

 private:
  int batch_size_;
  int height_;
  int width_;
  int channels_;
  int ele_count_;

  // Forward input
  framework::Tensor cpu_x_;
  framework::Tensor cpu_sum_;
  framework::Tensor cpu_sum_of_square_;
  framework::Tensor cpu_bn_scale_;
  framework::Tensor cpu_bn_bias_;

  double eps_ = 1e-5;
  float momentum_ = 0.9;
};

TEST(CudnnBNAddReluForward, GPUCudnnBNAddReluForwardFp16) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int channels = 64;
  FLAGS_cudnn_batchnorm_spatial_persistent = true;
  CudnnBNAddReluTester<paddle::platform::float16> test(batch_size, height,
                                                       width, channels);
  test.CheckForward(1e-3, true);
}
