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
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
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
  LOG(INFO) << "[" << name << "] The dims is [" << cpu_res.dims()
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

template <typename T>
void ComputeInplaceAdd(const framework::Tensor &cpu_x,
                       framework::Tensor *cpu_y) {
  EXPECT_EQ(cpu_x.dims(), cpu_y->dims());

  const T *cpu_x_ptr = cpu_x.data<T>();
  T *cpu_y_ptr = cpu_y->data<T>();
  for (int64_t i = 0; i < cpu_x.numel(); ++i) {
    cpu_y_ptr[i] += cpu_x_ptr[i];
  }
}

template <typename T>
void ComputeInplaceRelu(framework::Tensor *cpu_x) {
  T *cpu_x_ptr = cpu_x->data<T>();
  for (int64_t i = 0; i < cpu_x->numel(); ++i) {
    cpu_x_ptr[i] =
        cpu_x_ptr[i] > static_cast<T>(0) ? cpu_x_ptr[i] : static_cast<T>(0);
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
  CudnnBNAddReluTester(int batch_size, int height, int width, int channels,
                       std::string act_type, bool fuse_add, bool has_shortcut) {
    batch_size_ = batch_size;
    height_ = height;
    width_ = width;
    channels_ = channels;
    ele_count_ = batch_size_ * height_ * width_;
    act_type_ = act_type;
    fuse_add_ = fuse_add;
    has_shortcut_ = has_shortcut;
    SetUp();
  }

  ~CudnnBNAddReluTester() {}

  void CheckForward(float diff, bool is_relative_atol = false) {
    LOG(INFO) << "[CheckForward, diff=" << diff
              << ", is_relative_atol=" << is_relative_atol
              << "] act_type=" << act_type_ << ", fuse_add=" << fuse_add_
              << ", has_shortcut=" << has_shortcut_;
    platform::CUDADeviceContext *ctx =
        static_cast<platform::CUDADeviceContext *>(
            platform::DeviceContextPool::Instance().Get(
                platform::CUDAPlace(0)));

    auto select = [&](Tensor *in) { return has_shortcut_ ? in : nullptr; };

    framework::Tensor cpu_mean_base_x;
    framework::Tensor cpu_var_base_x;
    framework::Tensor cpu_saved_mean_base_x;
    framework::Tensor cpu_saved_var_base_x;
    framework::Tensor cpu_y_base;
    framework::Tensor cpu_reserve_space_base_x;
    framework::Tensor cpu_mean_base_z;
    framework::Tensor cpu_var_base_z;
    framework::Tensor cpu_saved_mean_base_z;
    framework::Tensor cpu_saved_var_base_z;
    framework::Tensor cpu_reserve_space_base_z;
    BaselineForward(*ctx, &cpu_mean_base_x, &cpu_var_base_x,
                    &cpu_saved_mean_base_x, &cpu_saved_var_base_x, &cpu_y_base,
                    &cpu_reserve_space_base_x, select(&cpu_mean_base_z),
                    select(&cpu_var_base_z), select(&cpu_saved_mean_base_z),
                    select(&cpu_saved_var_base_z),
                    select(&cpu_reserve_space_base_z));

    framework::Tensor cpu_mean_x;
    framework::Tensor cpu_var_x;
    framework::Tensor cpu_saved_mean_x;
    framework::Tensor cpu_saved_var_x;
    framework::Tensor cpu_y;
    framework::Tensor cpu_bitmask;
    framework::Tensor cpu_mean_z;
    framework::Tensor cpu_var_z;
    framework::Tensor cpu_saved_mean_z;
    framework::Tensor cpu_saved_var_z;
    FusedForward(*ctx, &cpu_mean_x, &cpu_var_x, &cpu_saved_mean_x,
                 &cpu_saved_var_x, &cpu_y, &cpu_bitmask, select(&cpu_mean_z),
                 select(&cpu_var_z), select(&cpu_saved_mean_z),
                 select(&cpu_saved_var_z));

    CheckOutput<float>("Mean", cpu_mean_x, cpu_mean_base_x, diff,
                       is_relative_atol);
    CheckOutput<float>("Variance", cpu_var_x, cpu_var_base_x, diff,
                       is_relative_atol);
    CheckOutput<float>("SavedMean", cpu_saved_mean_x, cpu_saved_mean_base_x,
                       diff, is_relative_atol);
    CheckOutput<float>("SavedVariance", cpu_saved_var_x, cpu_saved_var_base_x,
                       diff, is_relative_atol);
    if (has_shortcut_) {
      CheckOutput<float>("MeanZ", cpu_mean_z, cpu_mean_base_z, diff,
                         is_relative_atol);
      CheckOutput<float>("VarianceZ", cpu_var_z, cpu_var_base_z, diff,
                         is_relative_atol);
      CheckOutput<float>("SavedMeanZ", cpu_saved_mean_z, cpu_saved_mean_base_z,
                         diff, is_relative_atol);
      CheckOutput<float>("SavedVarianceZ", cpu_saved_var_z,
                         cpu_saved_var_base_z, diff, is_relative_atol);
    }
    CheckOutput<T>("Y", cpu_y, cpu_y_base, diff, is_relative_atol);
  }

 private:
  void SetUp() {
    // Initialize input data
    InitRandomTensor<T>({batch_size_, height_, width_, channels_}, &cpu_x_);
    InitRandomTensor<float>({channels_}, &cpu_bn_scale_x_);
    InitRandomTensor<float>({channels_}, &cpu_bn_bias_x_);

    if (has_shortcut_) {
      InitRandomTensor<T>({batch_size_, height_, width_, channels_}, &cpu_z_);
      InitRandomTensor<float>({channels_}, &cpu_bn_scale_z_);
      InitRandomTensor<float>({channels_}, &cpu_bn_bias_z_);
    } else {
      if (fuse_add_) {
        InitRandomTensor<T>({batch_size_, height_, width_, channels_}, &cpu_z_);
      }
    }
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

  void BaselineForward(const platform::CUDADeviceContext &ctx,
                       Tensor *cpu_mean_x, Tensor *cpu_var_x,
                       Tensor *cpu_saved_mean_x, Tensor *cpu_saved_var_x,
                       Tensor *cpu_y, Tensor *cpu_reserve_space_x,
                       Tensor *cpu_mean_z = nullptr,
                       Tensor *cpu_var_z = nullptr,
                       Tensor *cpu_saved_mean_z = nullptr,
                       Tensor *cpu_saved_var_z = nullptr,
                       Tensor *cpu_reserve_space_z = nullptr) {
    InitMeanVar(cpu_mean_x, cpu_var_x, cpu_saved_mean_x, cpu_saved_var_x);
    ComputeBatchNormForward(ctx, cpu_x_, cpu_bn_scale_x_, cpu_bn_bias_x_,
                            cpu_mean_x, cpu_var_x, cpu_saved_mean_x,
                            cpu_saved_var_x, cpu_y, cpu_reserve_space_x);
    if (has_shortcut_) {
      framework::Tensor cpu_z_out;
      InitMeanVar(cpu_mean_z, cpu_var_z, cpu_saved_mean_z, cpu_saved_var_z);
      ComputeBatchNormForward(ctx, cpu_z_, cpu_bn_scale_z_, cpu_bn_bias_z_,
                              cpu_mean_z, cpu_var_z, cpu_saved_mean_z,
                              cpu_saved_var_z, &cpu_z_out, cpu_reserve_space_z);
      ComputeInplaceAdd<T>(cpu_z_out, cpu_y);
    } else {
      if (fuse_add_) {
        ComputeInplaceAdd<T>(cpu_z_, cpu_y);
      }
    }
    if (act_type_ == "relu") {
      ComputeInplaceRelu<T>(cpu_y);
    }
  }

  void ComputeFusedBNStatsFinalize(const platform::CUDADeviceContext &ctx,
                                   const Tensor &cpu_x,
                                   const Tensor &cpu_bn_scale,
                                   const Tensor &cpu_bn_bias, Tensor *sum,
                                   Tensor *sum_of_square, Tensor *bn_scale,
                                   Tensor *bn_bias, Tensor *mean, Tensor *var,
                                   Tensor *saved_mean, Tensor *saved_var,
                                   Tensor *equiv_scale, Tensor *equiv_bias) {
    framework::Tensor cpu_sum;
    framework::Tensor cpu_sum_of_square;
    ComputeSumAndSquareSum<T>(cpu_x, &cpu_sum, &cpu_sum_of_square);

    auto place = ctx.GetPlace();
    TensorCopySync(cpu_sum, place, sum);
    TensorCopySync(cpu_sum_of_square, place, sum_of_square);
    TensorCopySync(cpu_bn_scale, place, bn_scale);
    TensorCopySync(cpu_bn_bias, place, bn_bias);

    bn_scale->Resize({1, 1, 1, channels_});
    bn_bias->Resize({1, 1, 1, channels_});

    // input
    float *sum_ptr = sum->data<float>();
    float *sum_of_square_ptr = sum_of_square->data<float>();
    float *bn_scale_ptr = bn_scale->data<float>();
    float *bn_bias_ptr = bn_bias->data<float>();

    mean->Resize({1, 1, 1, channels_});
    var->Resize({1, 1, 1, channels_});

    // output
    float *mean_ptr = mean->data<float>();
    float *var_ptr = var->data<float>();
    float *saved_mean_ptr =
        saved_mean->mutable_data<float>({1, 1, 1, channels_}, place);
    float *saved_var_ptr =
        saved_var->mutable_data<float>({1, 1, 1, channels_}, place);
    T *equiv_scale_ptr =
        equiv_scale->mutable_data<T>({1, 1, 1, channels_}, place);
    T *equiv_bias_ptr =
        equiv_bias->mutable_data<T>({1, 1, 1, channels_}, place);

    auto param_shape = framework::vectorize<int>(bn_scale->dims());
    op::CudnnBNStatsFinalize<T> bn_op(ctx, param_shape);
    bn_op.Forward(ctx, sum_ptr, sum_of_square_ptr, bn_scale_ptr, bn_bias_ptr,
                  saved_mean_ptr, saved_var_ptr, mean_ptr, var_ptr,
                  equiv_scale_ptr, equiv_bias_ptr, eps_, momentum_, ele_count_,
                  true);
  }

  // Get forward results of CudnnBNStatsFinalize + CudnnScaleBiasAddRelu
  void FusedForward(const platform::CUDADeviceContext &ctx, Tensor *cpu_mean_x,
                    Tensor *cpu_var_x, Tensor *cpu_saved_mean_x,
                    Tensor *cpu_saved_var_x, Tensor *cpu_y, Tensor *cpu_bitmask,
                    Tensor *cpu_mean_z = nullptr, Tensor *cpu_var_z = nullptr,
                    Tensor *cpu_saved_mean_z = nullptr,
                    Tensor *cpu_saved_var_z = nullptr) {
    framework::Tensor x;
    framework::Tensor sum_x;
    framework::Tensor sum_of_square_x;
    framework::Tensor bn_scale_x;
    framework::Tensor bn_bias_x;

    framework::Tensor z;
    framework::Tensor sum_z;
    framework::Tensor sum_of_square_z;
    framework::Tensor bn_scale_z;
    framework::Tensor bn_bias_z;

    auto place = ctx.GetPlace();
    TensorCopySync(cpu_x_, place, &x);
    if (fuse_add_ || has_shortcut_) {
      TensorCopySync(cpu_z_, place, &z);
    }

    framework::Tensor mean_x;
    framework::Tensor var_x;
    framework::Tensor saved_mean_x;
    framework::Tensor saved_var_x;
    framework::Tensor equiv_scale_x;
    framework::Tensor equiv_bias_x;

    framework::Tensor mean_z;
    framework::Tensor var_z;
    framework::Tensor saved_mean_z;
    framework::Tensor saved_var_z;
    framework::Tensor equiv_scale_z;
    framework::Tensor equiv_bias_z;

    framework::Tensor y;
    framework::Tensor bitmask;

    InitMeanVar(cpu_mean_x, cpu_var_x, cpu_saved_mean_x, cpu_saved_var_x);
    TensorCopySync(*cpu_mean_x, place, &mean_x);
    TensorCopySync(*cpu_var_x, place, &var_x);
    if (has_shortcut_) {
      InitMeanVar(cpu_mean_z, cpu_var_z, cpu_saved_mean_z, cpu_saved_var_z);
      TensorCopySync(*cpu_mean_z, place, &mean_z);
      TensorCopySync(*cpu_var_z, place, &var_z);
    }

    // 1. BN Stats Finalize
    ComputeFusedBNStatsFinalize(ctx, cpu_x_, cpu_bn_scale_x_, cpu_bn_bias_x_,
                                &sum_x, &sum_of_square_x, &bn_scale_x,
                                &bn_bias_x, &mean_x, &var_x, &saved_mean_x,
                                &saved_var_x, &equiv_scale_x, &equiv_bias_x);
    if (has_shortcut_) {
      ComputeFusedBNStatsFinalize(ctx, cpu_z_, cpu_bn_scale_z_, cpu_bn_bias_z_,
                                  &sum_z, &sum_of_square_z, &bn_scale_z,
                                  &bn_bias_z, &mean_z, &var_z, &saved_mean_z,
                                  &saved_var_z, &equiv_scale_z, &equiv_bias_z);
    }

    T *x_ptr = x.data<T>();
    T *z_ptr = (fuse_add_ || has_shortcut_) ? z.data<T>() : nullptr;
    T *equiv_scale_x_ptr = equiv_scale_x.data<T>();
    T *equiv_bias_x_ptr = equiv_bias_x.data<T>();
    T *equiv_scale_z_ptr = has_shortcut_ ? equiv_scale_z.data<T>() : nullptr;
    T *equiv_bias_z_ptr = has_shortcut_ ? equiv_bias_z.data<T>() : nullptr;
    T *y_ptr =
        y.mutable_data<T>({batch_size_, height_, width_, channels_}, place);

    int c = channels_;
    int64_t nhw = ele_count_;
    int32_t c_int32_elems = ((c + 63) & ~63) / 32;
    int32_t nhw_int32_elems = (nhw + 31) & ~31;
    int32_t *bitmask_ptr = bitmask.mutable_data<int32_t>(
        {nhw_int32_elems, c_int32_elems, 1}, place);

    auto data_shape = framework::vectorize<int>(x.dims());
    auto param_shape = framework::vectorize<int>(bn_scale_x.dims());
    auto bitmask_shape = framework::vectorize<int>(bitmask.dims());

    // 2. Scale Bias + Relu
    op::CudnnScaleBiasAddRelu<T> sbar_op(ctx, act_type_, fuse_add_,
                                         has_shortcut_, data_shape, param_shape,
                                         bitmask_shape);
    sbar_op.Forward(ctx, x_ptr, equiv_scale_x_ptr, equiv_bias_x_ptr, y_ptr,
                    bitmask_ptr, z_ptr, equiv_scale_z_ptr, equiv_bias_z_ptr);

    TensorCopySync(mean_x, platform::CPUPlace(), cpu_mean_x);
    TensorCopySync(var_x, platform::CPUPlace(), cpu_var_x);
    TensorCopySync(saved_mean_x, platform::CPUPlace(), cpu_saved_mean_x);
    TensorCopySync(saved_var_x, platform::CPUPlace(), cpu_saved_var_x);
    if (has_shortcut_) {
      TensorCopySync(mean_z, platform::CPUPlace(), cpu_mean_z);
      TensorCopySync(var_z, platform::CPUPlace(), cpu_var_z);
      TensorCopySync(saved_mean_z, platform::CPUPlace(), cpu_saved_mean_z);
      TensorCopySync(saved_var_z, platform::CPUPlace(), cpu_saved_var_z);
    }
    TensorCopySync(y, platform::CPUPlace(), cpu_y);
    TensorCopySync(bitmask, platform::CPUPlace(), cpu_bitmask);
  }

 private:
  int batch_size_;
  int height_;
  int width_;
  int channels_;
  int ele_count_;

  std::string act_type_;
  bool fuse_add_;
  bool has_shortcut_;

  // Forward input
  framework::Tensor cpu_x_;
  framework::Tensor cpu_bn_scale_x_;
  framework::Tensor cpu_bn_bias_x_;
  framework::Tensor cpu_z_;
  framework::Tensor cpu_bn_scale_z_;
  framework::Tensor cpu_bn_bias_z_;

  double eps_ = 1e-5;
  float momentum_ = 0.9;
};

TEST(CudnnBNAddReluFp16, BNAdd) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int channels = 64;
  std::string act_type = "";
  bool has_shortcut = false;
  FLAGS_cudnn_batchnorm_spatial_persistent = true;
  for (auto fuse_add : {false, true}) {
    CudnnBNAddReluTester<paddle::platform::float16> test(
        batch_size, height, width, channels, act_type, fuse_add, has_shortcut);
    test.CheckForward(2e-3);
  }
}

TEST(CudnnBNAddReluFp16, BNAddRelu) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int channels = 64;
  std::string act_type = "relu";
  bool has_shortcut = false;
  FLAGS_cudnn_batchnorm_spatial_persistent = true;
  for (auto fuse_add : {false, true}) {
    CudnnBNAddReluTester<paddle::platform::float16> test(
        batch_size, height, width, channels, act_type, fuse_add, has_shortcut);
    test.CheckForward(2e-3);
  }
}

TEST(CudnnBNAddReluFp16, HasShortcut) {
  int batch_size = 4;
  int height = 8;
  int width = 8;
  int channels = 64;
  std::string act_type = "";
  bool fuse_add = false;
  bool has_shortcut = true;
  FLAGS_cudnn_batchnorm_spatial_persistent = true;
  CudnnBNAddReluTester<paddle::platform::float16> test(
      batch_size, height, width, channels, act_type, fuse_add, has_shortcut);
  test.CheckForward(5e-3);
}
