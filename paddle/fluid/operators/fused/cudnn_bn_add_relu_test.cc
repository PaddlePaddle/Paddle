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
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/kernels/funcs/math_function.h"

DECLARE_bool(cudnn_batchnorm_spatial_persistent);

namespace framework = paddle::framework;
namespace platform = paddle::platform;
namespace op = paddle::operators;
using Tensor = paddle::framework::Tensor;

USE_OP_ITSELF(batch_norm);
USE_CUDA_ONLY_OP(fused_bn_add_activation);
USE_CUDA_ONLY_OP(fused_bn_add_activation_grad);

template <typename T>
void InitRandomTensor(const std::vector<int64_t> &dims,
                      framework::Tensor *cpu_out) {
  T *cpu_out_ptr =
      cpu_out->mutable_data<T>(phi::make_ddim(dims), platform::CPUPlace());
  std::default_random_engine random(0);
  std::uniform_real_distribution<float> dis(-1.0, 1.0);
  for (int i = 0; i < cpu_out->numel(); ++i) {
    cpu_out_ptr[i] = static_cast<T>(dis(random));
  }
}

template <typename T>
void InitConstantTensor(const std::vector<int64_t> &dims, T value,
                        framework::Tensor *cpu_out) {
  T *cpu_out_ptr =
      cpu_out->mutable_data<T>(phi::make_ddim(dims), platform::CPUPlace());
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

void ComputeBatchNormForward(const platform::CUDADeviceContext &ctx,
                             const Tensor &cpu_x, const Tensor &cpu_scale,
                             const Tensor &cpu_bias, Tensor *cpu_mean,
                             Tensor *cpu_var, Tensor *cpu_saved_mean,
                             Tensor *cpu_saved_var, Tensor *cpu_y,
                             Tensor *saved_reserve_space) {
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
  paddle::framework::TensorCopySync(cpu_x, place, x);
  paddle::framework::TensorCopySync(cpu_scale, place, scale);
  paddle::framework::TensorCopySync(cpu_bias, place, bias);
  paddle::framework::TensorCopySync(*cpu_mean, place, mean);
  paddle::framework::TensorCopySync(*cpu_var, place, var);

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

  paddle::framework::TensorCopySync(*y, platform::CPUPlace(), cpu_y);
  paddle::framework::TensorCopySync(*mean, platform::CPUPlace(), cpu_mean);
  paddle::framework::TensorCopySync(*var, platform::CPUPlace(), cpu_var);
  paddle::framework::TensorCopySync(*saved_mean, platform::CPUPlace(),
                                    cpu_saved_mean);
  paddle::framework::TensorCopySync(*saved_var, platform::CPUPlace(),
                                    cpu_saved_var);
  // reserved_space will stay on GPU and used in grad op.
  saved_reserve_space->ShareDataWith(*reserve_space);
}

void ComputeFusedBNAddReluForward(const platform::CUDADeviceContext &ctx,
                                  const Tensor &cpu_x, const Tensor &cpu_z,
                                  const Tensor &cpu_scale,
                                  const Tensor &cpu_bias, Tensor *cpu_mean,
                                  Tensor *cpu_var, Tensor *cpu_saved_mean,
                                  Tensor *cpu_saved_var, Tensor *cpu_y,
                                  Tensor *saved_reserve_space) {
  framework::Scope scope;
  auto *x = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("Z")->GetMutable<framework::LoDTensor>();
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
  paddle::framework::TensorCopySync(cpu_x, place, x);
  paddle::framework::TensorCopySync(cpu_z, place, z);
  paddle::framework::TensorCopySync(cpu_scale, place, scale);
  paddle::framework::TensorCopySync(cpu_bias, place, bias);
  paddle::framework::TensorCopySync(*cpu_mean, place, mean);
  paddle::framework::TensorCopySync(*cpu_var, place, var);

  int64_t channels = x->dims()[3];
  scale->Resize({channels});
  bias->Resize({channels});
  mean->Resize({channels});
  var->Resize({channels});

  framework::AttributeMap attrs;

  auto op = framework::OpRegistry::CreateOp(
      "fused_bn_add_activation",
      {{"X", {"X"}}, {"Z", {"Z"}}, {"Scale", {"Scale"}}, {"Bias", {"Bias"}}},
      {{"Y", {"Y"}},
       {"MeanOut", {"Mean"}},
       {"VarianceOut", {"Variance"}},
       {"SavedMean", {"SavedMean"}},
       {"SavedVariance", {"SavedVariance"}},
       {"ReserveSpace", {"ReserveSpace"}}},
      attrs);
  op->Run(scope, ctx.GetPlace());

  paddle::framework::TensorCopySync(*y, platform::CPUPlace(), cpu_y);
  paddle::framework::TensorCopySync(*mean, platform::CPUPlace(), cpu_mean);
  paddle::framework::TensorCopySync(*var, platform::CPUPlace(), cpu_var);
  paddle::framework::TensorCopySync(*saved_mean, platform::CPUPlace(),
                                    cpu_saved_mean);
  paddle::framework::TensorCopySync(*saved_var, platform::CPUPlace(),
                                    cpu_saved_var);
  // reserved_space will stay on GPU and used in grad op.
  saved_reserve_space->ShareDataWith(*reserve_space);
}

void ComputeFusedBNAddReluBackward(
    const platform::CUDADeviceContext &ctx, const Tensor &cpu_dy,
    const Tensor &cpu_x, const Tensor &cpu_scale, const Tensor &cpu_bias,
    const Tensor &cpu_saved_mean, const Tensor &cpu_saved_var,
    const Tensor &cpu_y, const Tensor &saved_reserve_space, Tensor *cpu_dx,
    Tensor *cpu_dz, Tensor *cpu_dscale, Tensor *cpu_dbias) {
  framework::Scope scope;
  auto *x = scope.Var("X")->GetMutable<framework::LoDTensor>();
  auto *y = scope.Var("Y")->GetMutable<framework::LoDTensor>();
  auto *dy = scope.Var("Y@GRAD")->GetMutable<framework::LoDTensor>();
  auto *scale = scope.Var("Scale")->GetMutable<framework::LoDTensor>();
  auto *bias = scope.Var("Bias")->GetMutable<framework::LoDTensor>();
  auto *saved_mean = scope.Var("SavedMean")->GetMutable<framework::LoDTensor>();
  auto *saved_var =
      scope.Var("SavedVariance")->GetMutable<framework::LoDTensor>();
  auto *reserve_space =
      scope.Var("ReserveSpace")->GetMutable<framework::LoDTensor>();
  auto *dx = scope.Var("X@GRAD")->GetMutable<framework::LoDTensor>();
  auto *dz = scope.Var("Z@GRAD")->GetMutable<framework::LoDTensor>();
  auto *dscale = scope.Var("Scale@GRAD")->GetMutable<framework::LoDTensor>();
  auto *dbias = scope.Var("Bias@GRAD")->GetMutable<framework::LoDTensor>();

  auto place = ctx.GetPlace();
  paddle::framework::TensorCopySync(cpu_x, place, x);
  paddle::framework::TensorCopySync(cpu_y, place, y);
  paddle::framework::TensorCopySync(cpu_dy, place, dy);
  paddle::framework::TensorCopySync(cpu_scale, place, scale);
  paddle::framework::TensorCopySync(cpu_bias, place, bias);
  paddle::framework::TensorCopySync(cpu_saved_mean, place, saved_mean);
  paddle::framework::TensorCopySync(cpu_saved_var, place, saved_var);
  reserve_space->ShareDataWith(saved_reserve_space);

  int64_t channels = x->dims()[3];
  scale->Resize({channels});
  bias->Resize({channels});
  saved_mean->Resize({channels});
  saved_var->Resize({channels});

  framework::AttributeMap attrs;
  float momentum = 0.9;
  float epsilon = 1e-5;
  std::string act_type = "relu";
  attrs.insert({"momentum", momentum});
  attrs.insert({"epsilon", epsilon});
  attrs.insert({"act_type", act_type});

  auto op = framework::OpRegistry::CreateOp(
      "fused_bn_add_activation_grad", {{"X", {"X"}},
                                       {"Y", {"Y"}},
                                       {"Y@GRAD", {"Y@GRAD"}},
                                       {"Scale", {"Scale"}},
                                       {"Bias", {"Bias"}},
                                       {"SavedMean", {"SavedMean"}},
                                       {"SavedVariance", {"SavedVariance"}},
                                       {"ReserveSpace", {"ReserveSpace"}}},
      {{"X@GRAD", {"X@GRAD"}},
       {"Z@GRAD", {"Z@GRAD"}},
       {"Scale@GRAD", {"Scale@GRAD"}},
       {"Bias@GRAD", {"Bias@GRAD"}}},
      attrs);
  op->Run(scope, ctx.GetPlace());

  paddle::framework::TensorCopySync(*dx, platform::CPUPlace(), cpu_dx);
  paddle::framework::TensorCopySync(*dz, platform::CPUPlace(), cpu_dz);
  paddle::framework::TensorCopySync(*dscale, platform::CPUPlace(), cpu_dscale);
  paddle::framework::TensorCopySync(*dbias, platform::CPUPlace(), cpu_dbias);
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
    framework::Tensor cpu_mean_base_z;
    framework::Tensor cpu_var_base_z;
    if (!has_shortcut_ && fuse_add_ && (act_type_ == "relu")) {
      BaselineForwardFusedBNAddRelu(
          *ctx, &cpu_mean_base_x, &cpu_var_base_x, &cpu_saved_mean_base_x_,
          &cpu_saved_var_base_x_, &cpu_y_base_, &saved_reserve_space_x_);
    } else {
      BaselineForward(
          *ctx, &cpu_mean_base_x, &cpu_var_base_x, &cpu_saved_mean_base_x_,
          &cpu_saved_var_base_x_, &cpu_y_base_, &saved_reserve_space_x_,
          select(&cpu_mean_base_z), select(&cpu_var_base_z),
          select(&cpu_saved_mean_base_z_), select(&cpu_saved_var_base_z_),
          select(&saved_reserve_space_z_));
    }

    framework::Tensor cpu_mean_x;
    framework::Tensor cpu_var_x;
    framework::Tensor cpu_y;
    framework::Tensor cpu_mean_z;
    framework::Tensor cpu_var_z;
    FusedForward(*ctx, &cpu_mean_x, &cpu_var_x, &cpu_saved_mean_x_,
                 &cpu_saved_var_x_, &cpu_y, &cpu_bitmask_, select(&cpu_mean_z),
                 select(&cpu_var_z), select(&cpu_saved_mean_z_),
                 select(&cpu_saved_var_z_));

    CheckOutput<float>("Mean", cpu_mean_x, cpu_mean_base_x, diff,
                       is_relative_atol);
    CheckOutput<float>("Variance", cpu_var_x, cpu_var_base_x, diff,
                       is_relative_atol);
    CheckOutput<float>("SavedMean", cpu_saved_mean_x_, cpu_saved_mean_base_x_,
                       diff, is_relative_atol);
    CheckOutput<float>("SavedVariance", cpu_saved_var_x_, cpu_saved_var_base_x_,
                       diff, is_relative_atol);
    if (has_shortcut_) {
      CheckOutput<float>("MeanZ", cpu_mean_z, cpu_mean_base_z, diff,
                         is_relative_atol);
      CheckOutput<float>("VarianceZ", cpu_var_z, cpu_var_base_z, diff,
                         is_relative_atol);
      CheckOutput<float>("SavedMeanZ", cpu_saved_mean_z_,
                         cpu_saved_mean_base_z_, diff, is_relative_atol);
      CheckOutput<float>("SavedVarianceZ", cpu_saved_var_z_,
                         cpu_saved_var_base_z_, diff, is_relative_atol);
    }
    CheckOutput<T>("Y", cpu_y, cpu_y_base_, diff, is_relative_atol);
  }

  void CheckBackward(float diff, bool is_relative_atol = false) {
    platform::CUDADeviceContext *ctx =
        static_cast<platform::CUDADeviceContext *>(
            platform::DeviceContextPool::Instance().Get(
                platform::CUDAPlace(0)));

    framework::Tensor cpu_dx_base;
    framework::Tensor cpu_dz_base;
    framework::Tensor cpu_dscale_base;
    framework::Tensor cpu_dbias_base;
    BaselineBackwardFusedBNAddRelu(*ctx, &cpu_dx_base, &cpu_dz_base,
                                   &cpu_dscale_base, &cpu_dbias_base);

    framework::Tensor cpu_dx;
    framework::Tensor cpu_dz;
    framework::Tensor cpu_dscale;
    framework::Tensor cpu_dbias;
    FusedBackward(*ctx, &cpu_dx, &cpu_dz, &cpu_dscale, &cpu_dbias);

    CheckOutput<T>("DX", cpu_dx, cpu_dx_base, diff, is_relative_atol);
    CheckOutput<T>("DZ", cpu_dz, cpu_dz_base, diff, is_relative_atol);
    CheckOutput<float>("DScale", cpu_dscale, cpu_dscale_base, diff,
                       is_relative_atol);
    CheckOutput<float>("DBias", cpu_dbias, cpu_dbias_base, diff,
                       is_relative_atol);
  }

 private:
  void SetUp() {
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

    InitRandomTensor<T>({batch_size_, height_, width_, channels_}, &cpu_dy_);
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
                       Tensor *cpu_y, Tensor *saved_reserve_space_x,
                       Tensor *cpu_mean_z = nullptr,
                       Tensor *cpu_var_z = nullptr,
                       Tensor *cpu_saved_mean_z = nullptr,
                       Tensor *cpu_saved_var_z = nullptr,
                       Tensor *saved_reserve_space_z = nullptr) {
    InitMeanVar(cpu_mean_x, cpu_var_x, cpu_saved_mean_x, cpu_saved_var_x);
    ComputeBatchNormForward(ctx, cpu_x_, cpu_bn_scale_x_, cpu_bn_bias_x_,
                            cpu_mean_x, cpu_var_x, cpu_saved_mean_x,
                            cpu_saved_var_x, cpu_y, saved_reserve_space_x);
    if (has_shortcut_) {
      framework::Tensor cpu_z_out;
      InitMeanVar(cpu_mean_z, cpu_var_z, cpu_saved_mean_z, cpu_saved_var_z);
      ComputeBatchNormForward(
          ctx, cpu_z_, cpu_bn_scale_z_, cpu_bn_bias_z_, cpu_mean_z, cpu_var_z,
          cpu_saved_mean_z, cpu_saved_var_z, &cpu_z_out, saved_reserve_space_z);
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

  void BaselineForwardFusedBNAddRelu(const platform::CUDADeviceContext &ctx,
                                     Tensor *cpu_mean, Tensor *cpu_var,
                                     Tensor *cpu_saved_mean,
                                     Tensor *cpu_saved_var, Tensor *cpu_y,
                                     Tensor *saved_reserve_space) {
    InitMeanVar(cpu_mean, cpu_var, cpu_saved_mean, cpu_saved_var);
    ComputeFusedBNAddReluForward(
        ctx, cpu_x_, cpu_z_, cpu_bn_scale_x_, cpu_bn_bias_x_, cpu_mean, cpu_var,
        cpu_saved_mean, cpu_saved_var, cpu_y, saved_reserve_space);
  }

  void BaselineBackwardFusedBNAddRelu(const platform::CUDADeviceContext &ctx,
                                      Tensor *cpu_dx, Tensor *cpu_dz,
                                      Tensor *cpu_dscale, Tensor *cpu_dbias) {
    ComputeFusedBNAddReluBackward(
        ctx, cpu_dy_, cpu_x_, cpu_bn_scale_x_, cpu_bn_bias_x_,
        cpu_saved_mean_base_x_, cpu_saved_var_base_x_, cpu_y_base_,
        saved_reserve_space_x_, cpu_dx, cpu_dz, cpu_dscale, cpu_dbias);
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
    paddle::framework::TensorCopySync(cpu_sum, place, sum);
    paddle::framework::TensorCopySync(cpu_sum_of_square, place, sum_of_square);
    paddle::framework::TensorCopySync(cpu_bn_scale, place, bn_scale);
    paddle::framework::TensorCopySync(cpu_bn_bias, place, bn_bias);

    bn_scale->Resize({1, 1, 1, channels_});
    bn_bias->Resize({1, 1, 1, channels_});

    // input
    mean->Resize({1, 1, 1, channels_});
    var->Resize({1, 1, 1, channels_});

    // output
    equiv_scale->Resize({1, 1, 1, channels_});
    equiv_bias->Resize({1, 1, 1, channels_});
    saved_mean->Resize({1, 1, 1, channels_});
    saved_var->Resize({1, 1, 1, channels_});

    auto param_shape = phi::vectorize<int>(bn_scale->dims());
    op::CudnnBNStatsFinalize<T> bn_op(ctx, param_shape);
    bn_op.Forward(ctx, *sum, *sum_of_square, *bn_scale, *bn_bias, saved_mean,
                  saved_var, mean, var, equiv_scale, equiv_bias, eps_,
                  momentum_, ele_count_, true);
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
    paddle::framework::TensorCopySync(cpu_x_, place, &x);
    if (fuse_add_ || has_shortcut_) {
      paddle::framework::TensorCopySync(cpu_z_, place, &z);
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
    paddle::framework::TensorCopySync(*cpu_mean_x, place, &mean_x);
    paddle::framework::TensorCopySync(*cpu_var_x, place, &var_x);
    if (has_shortcut_) {
      InitMeanVar(cpu_mean_z, cpu_var_z, cpu_saved_mean_z, cpu_saved_var_z);
      paddle::framework::TensorCopySync(*cpu_mean_z, place, &mean_z);
      paddle::framework::TensorCopySync(*cpu_var_z, place, &var_z);
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

    y.Resize(phi::make_ddim({batch_size_, height_, width_, channels_}));

    int c = channels_;
    int64_t nhw = ele_count_;
    int32_t c_int32_elems = ((c + 63) & ~63) / 32;
    int32_t nhw_int32_elems = (nhw + 31) & ~31;
    bitmask.Resize(phi::make_ddim({nhw_int32_elems, c_int32_elems, 1}));

    auto data_shape = phi::vectorize<int>(x.dims());
    auto param_shape = phi::vectorize<int>(bn_scale_x.dims());
    auto bitmask_shape = phi::vectorize<int>(bitmask.dims());

    // 2. Scale Bias + Relu
    op::CudnnScaleBiasAddRelu<T> sbar_op(ctx, act_type_, fuse_add_,
                                         has_shortcut_, data_shape, param_shape,
                                         bitmask_shape);
    sbar_op.Forward(ctx, x, equiv_scale_x, equiv_bias_x, &z, &equiv_scale_z,
                    &equiv_bias_z, &y, &bitmask);

    paddle::framework::TensorCopySync(mean_x, platform::CPUPlace(), cpu_mean_x);
    paddle::framework::TensorCopySync(var_x, platform::CPUPlace(), cpu_var_x);
    paddle::framework::TensorCopySync(saved_mean_x, platform::CPUPlace(),
                                      cpu_saved_mean_x);
    paddle::framework::TensorCopySync(saved_var_x, platform::CPUPlace(),
                                      cpu_saved_var_x);
    if (has_shortcut_) {
      paddle::framework::TensorCopySync(mean_z, platform::CPUPlace(),
                                        cpu_mean_z);
      paddle::framework::TensorCopySync(var_z, platform::CPUPlace(), cpu_var_z);
      paddle::framework::TensorCopySync(saved_mean_z, platform::CPUPlace(),
                                        cpu_saved_mean_z);
      paddle::framework::TensorCopySync(saved_var_z, platform::CPUPlace(),
                                        cpu_saved_var_z);
    }
    paddle::framework::TensorCopySync(y, platform::CPUPlace(), cpu_y);
    paddle::framework::TensorCopySync(bitmask, platform::CPUPlace(),
                                      cpu_bitmask);
  }

  // Get backward results of CudnnBNStatsFinalize + CudnnScaleBiasAddRelu
  void FusedBackward(const platform::CUDADeviceContext &ctx, Tensor *cpu_dx,
                     Tensor *cpu_dz, Tensor *cpu_dscale, Tensor *cpu_dbias) {
    framework::Tensor dy;
    framework::Tensor x;
    framework::Tensor bn_scale;
    framework::Tensor bn_bias;
    framework::Tensor saved_mean;
    framework::Tensor saved_var;
    framework::Tensor bitmask;
    framework::Tensor dx;
    framework::Tensor dz;
    framework::Tensor dscale;
    framework::Tensor dbias;

    auto place = ctx.GetPlace();
    paddle::framework::TensorCopySync(cpu_dy_, place, &dy);
    paddle::framework::TensorCopySync(cpu_x_, place, &x);
    paddle::framework::TensorCopySync(cpu_bn_scale_x_, place, &bn_scale);
    paddle::framework::TensorCopySync(cpu_bn_bias_x_, place, &bn_bias);
    paddle::framework::TensorCopySync(cpu_saved_mean_x_, place, &saved_mean);
    paddle::framework::TensorCopySync(cpu_saved_var_x_, place, &saved_var);
    paddle::framework::TensorCopySync(cpu_bitmask_, place, &bitmask);

    bn_scale.Resize({1, 1, 1, channels_});
    bn_bias.Resize({1, 1, 1, channels_});
    saved_mean.Resize({1, 1, 1, channels_});
    saved_var.Resize({1, 1, 1, channels_});

    dx.Resize(phi::make_ddim({batch_size_, height_, width_, channels_}));
    dz.Resize(phi::make_ddim({batch_size_, height_, width_, channels_}));
    dscale.Resize(phi::make_ddim({1, 1, 1, channels_}));
    dbias.Resize(phi::make_ddim({1, 1, 1, channels_}));

    auto data_shape = phi::vectorize<int>(x.dims());
    auto param_shape = phi::vectorize<int>(bn_scale.dims());
    auto bitmask_shape = phi::vectorize<int>(bitmask.dims());

    std::string act_type = "relu";
    op::CudnnScaleBiasAddRelu<T> sbar_op(ctx, act_type, true, false, data_shape,
                                         param_shape, bitmask_shape);
    sbar_op.Backward(ctx, dy, x, bn_scale, bn_bias, saved_mean, saved_var,
                     &bitmask, &dx, &dz, &dscale, &dbias, eps_);

    paddle::framework::TensorCopySync(dx, platform::CPUPlace(), cpu_dx);
    paddle::framework::TensorCopySync(dz, platform::CPUPlace(), cpu_dz);
    paddle::framework::TensorCopySync(dscale, platform::CPUPlace(), cpu_dscale);
    paddle::framework::TensorCopySync(dbias, platform::CPUPlace(), cpu_dbias);
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

  // Backward input
  framework::Tensor cpu_dy_;
  framework::Tensor cpu_bitmask_;
  framework::Tensor cpu_saved_mean_x_;
  framework::Tensor cpu_saved_var_x_;
  framework::Tensor cpu_saved_mean_z_;
  framework::Tensor cpu_saved_var_z_;
  framework::Tensor cpu_saved_mean_base_x_;
  framework::Tensor cpu_saved_var_base_x_;
  framework::Tensor saved_reserve_space_x_;
  framework::Tensor cpu_saved_mean_base_z_;
  framework::Tensor cpu_saved_var_base_z_;
  framework::Tensor saved_reserve_space_z_;
  framework::Tensor cpu_y_base_;

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
    if (fuse_add) {
      test.CheckBackward(2e-4);
    }
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
