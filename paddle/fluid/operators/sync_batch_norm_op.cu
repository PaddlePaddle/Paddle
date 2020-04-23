/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/sync_batch_norm_op.cu.h"

namespace paddle {
namespace operators {

template <typename T>
class SyncBatchNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const float momentum = ctx.Attr<float>("momentum");
    const bool is_test = ctx.Attr<bool>("is_test");
    const std::string layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout layout = framework::StringToDataLayout(layout_str);
    const bool use_global_stats = ctx.Attr<bool>("use_global_stats");
    const bool trainable_stats = ctx.Attr<bool>("trainable_statistics");
    PADDLE_ENFORCE_EQ(use_global_stats, false,
                      platform::errors::InvalidArgument(
                          "sync_batch_norm doesn't support "
                          "to set use_global_stats True. Please use batch_norm "
                          "in this case."));

    const auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");

    const auto *est_mean = ctx.Input<Tensor>("Mean");
    const auto *est_var = ctx.Input<Tensor>("Variance");

    // moving mean/variance
    auto *mean_out = ctx.Output<Tensor>("MeanOut");
    auto *variance_out = ctx.Output<Tensor>("VarianceOut");

    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_inv_variance = ctx.Output<Tensor>("SavedVariance");

    bool test_mode = is_test && (!trainable_stats);
    SyncBatchNormFunctor<platform::CUDADeviceContext, T>(
        ctx, layout, x, y, est_mean, est_var, mean_out, variance_out,
        saved_mean, saved_inv_variance, epsilon, momentum, test_mode,
        use_global_stats);
  }
};

template <typename T>
class SyncBatchNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument("It must use CUDAPlace."));
    double epsilon = static_cast<double>(ctx.Attr<float>("epsilon"));
    const std::string layout_str = ctx.Attr<std::string>("data_layout");

    const DataLayout layout = framework::StringToDataLayout(layout_str);
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    // init output
    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_inv_var = ctx.Input<Tensor>("SavedVariance");

    SyncBatchNormGradFunctor<platform::CUDADeviceContext, T>(
        ctx, layout, scale, bias, d_x, d_y, d_scale, d_bias, saved_mean,
        saved_inv_var, epsilon);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    sync_batch_norm, ops::SyncBatchNormKernel<plat::CUDADeviceContext, float>,
    ops::SyncBatchNormKernel<plat::CUDADeviceContext, double>,
    ops::SyncBatchNormKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    sync_batch_norm_grad,
    ops::SyncBatchNormGradKernel<plat::CUDADeviceContext, float>,
    ops::SyncBatchNormGradKernel<plat::CUDADeviceContext, double>,
    ops::SyncBatchNormGradKernel<plat::CUDADeviceContext, plat::float16>);

// clang-format on
