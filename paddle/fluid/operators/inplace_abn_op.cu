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

#include "paddle/fluid/operators/inplace_abn_op.h"
#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/phi/kernels/batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/batch_norm_kernel.h"
#include "paddle/phi/kernels/gpu/sync_batch_norm_utils.h"
#include "paddle/phi/kernels/sync_batch_norm_grad_kernel.h"
#include "paddle/phi/kernels/sync_batch_norm_kernel.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InplaceABNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* y = ctx.Output<phi::DenseTensor>("Y");
    auto* x = ctx.Input<phi::DenseTensor>("X");
    PADDLE_ENFORCE_EQ(x,
                      y,
                      platform::errors::InvalidArgument(
                          "X and Y not inplaced in inplace mode"));
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* mean = ctx.Input<phi::DenseTensor>("Mean");
    auto* variance = ctx.Input<phi::DenseTensor>("Variance");

    auto momentum = ctx.Attr<float>("momentum");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto data_layout = ctx.Attr<std::string>("data_layout");
    auto is_test = ctx.Attr<bool>("is_test");
    auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    auto trainable_statistics = ctx.Attr<bool>("trainable_statistics");

    auto* mean_out = ctx.Output<phi::DenseTensor>("MeanOut");
    auto* variance_out = ctx.Output<phi::DenseTensor>("VarianceOut");
    auto* saved_mean = ctx.Output<phi::DenseTensor>("SavedMean");
    auto* saved_variance = ctx.Output<phi::DenseTensor>("SavedVariance");
    auto* reserve_space = ctx.Output<phi::DenseTensor>("ReserveSpace");

    if (ctx.Attr<bool>("use_sync_bn")) {
      auto& dev_ctx = ctx.device_context<DeviceContext>();
      phi::SyncBatchNormKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *x,
          *mean,
          *variance,
          *scale,
          *bias,
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics,
          y,
          mean_out,
          variance_out,
          saved_mean,
          saved_variance,
          reserve_space);
    } else {
      auto& dev_ctx = ctx.device_context<DeviceContext>();
      phi::BatchNormKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *x,
          *mean,
          *variance,
          *scale,
          *bias,
          is_test,
          momentum,
          epsilon,
          data_layout,
          use_global_stats,
          trainable_statistics,
          y,
          mean_out,
          variance_out,
          saved_mean,
          saved_variance,
          reserve_space);
    }

    auto cur_y = EigenVector<T>::Flatten(*y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(ctx, activation, place, cur_y, cur_y);
  }
};

// Deriving the Gradient for the Backward Pass of Batch Normalization
// https://kevinzakka.github.io/2016/09/14/batch_normalization/
template <typename DeviceContext, typename T>
class InplaceABNGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* y = ctx.Input<phi::DenseTensor>("Y");
    auto* d_y = ctx.Input<phi::DenseTensor>(framework::GradVarName("Y"));
    auto* d_x = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    PADDLE_ENFORCE_EQ(d_x,
                      d_y,
                      platform::errors::InvalidArgument(
                          "X@GRAD and Y@GRAD not inplaced in inplace mode"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));

    auto py = *y;
    auto pd_y = *d_y;
    auto cur_y = EigenVector<T>::Flatten(py);
    auto cur_dy = EigenVector<T>::Flatten(pd_y);

    InplaceABNActivation<DeviceContext, T> functor;
    functor.GradCompute(ctx, activation, place, cur_y, cur_y, cur_dy, cur_dy);

    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* saved_mean = ctx.Input<phi::DenseTensor>("SavedMean");
    auto* saved_variance = ctx.Input<phi::DenseTensor>("SavedVariance");

    auto momentum = ctx.Attr<float>("momentum");
    auto epsilon = ctx.Attr<float>("epsilon");
    auto data_layout = ctx.Attr<std::string>("data_layout");
    auto is_test = ctx.Attr<bool>("is_test");
    auto use_global_stats = ctx.Attr<bool>("use_global_stats");
    auto trainable_statistics = ctx.Attr<bool>("trainable_statistics");

    auto* scale_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* bias_grad =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    auto* reserve_space = ctx.Input<phi::DenseTensor>("ReserveSpace");
    auto* mean = ctx.Input<phi::DenseTensor>("ReserveSpace");
    auto* variance = ctx.Input<phi::DenseTensor>("ReserveSpace");

    if (ctx.Attr<bool>("use_sync_bn")) {
      auto& dev_ctx = ctx.device_context<DeviceContext>();
      phi::SyncBatchNormGradFunctor<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          nullptr,
          y,
          *scale,
          *bias,
          *saved_mean,
          *saved_variance,
          *d_y,
          epsilon,
          data_layout,
          d_x,
          scale_grad,
          bias_grad);
    } else {
      paddle::optional<phi::DenseTensor> space_opt;
      paddle::optional<phi::DenseTensor> mean_opt;
      paddle::optional<phi::DenseTensor> variance_opt;

      if (reserve_space != nullptr) {
        space_opt = *reserve_space;
      }

      if (mean != nullptr) {
        mean_opt = *mean;
      }

      if (variance != nullptr) {
        variance_opt = *variance;
      }

      auto& dev_ctx = ctx.device_context<DeviceContext>();
      phi::BatchNormGradRawKernel<T>(
          static_cast<const typename framework::ConvertToPhiContext<
              DeviceContext>::TYPE&>(dev_ctx),
          *y,
          *scale,
          *bias,
          mean_opt,
          variance_opt,
          *saved_mean,
          *saved_variance,
          space_opt,
          *d_y,
          momentum,
          epsilon,
          data_layout,
          is_test,
          use_global_stats,
          trainable_statistics,
          true,
          d_x,
          scale_grad,
          bias_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
// MIOPEN do not support double
REGISTER_OP_CUDA_KERNEL(inplace_abn,
                        ops::InplaceABNKernel<phi::GPUContext, float>);
REGISTER_OP_CUDA_KERNEL(inplace_abn_grad,
                        ops::InplaceABNGradKernel<phi::GPUContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(inplace_abn,
                        ops::InplaceABNKernel<phi::GPUContext, float>,
                        ops::InplaceABNKernel<phi::GPUContext, double>);
REGISTER_OP_CUDA_KERNEL(inplace_abn_grad,
                        ops::InplaceABNGradKernel<phi::GPUContext, float>,
                        ops::InplaceABNGradKernel<phi::GPUContext, double>);
#endif
