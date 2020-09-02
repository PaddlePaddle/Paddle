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

#include "paddle/fluid/operators/batch_norm_op.h"
#include "paddle/fluid/operators/inplace_abn_op.h"
#include "paddle/fluid/operators/sync_batch_norm_op.cu.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InplaceABNKernel
    : public paddle::operators::SyncBatchNormKernel<DeviceContext, T>,
      public paddle::operators::BatchNormKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* y = ctx.Output<Tensor>("Y");
    auto* x = ctx.Input<Tensor>("X");
    PADDLE_ENFORCE_EQ(x, y, platform::errors::InvalidArgument(
                                "X and Y not inplaced in inplace mode"));
    auto activation =
        GetInplaceABNActivationType(ctx.Attr<std::string>("activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    if (ctx.Attr<bool>("use_sync_bn")) {
      SyncBatchNormKernel<DeviceContext, T>::Compute(ctx);
    } else {
      BatchNormKernel<DeviceContext, T>::Compute(ctx);
    }

    auto cur_y = EigenVector<T>::Flatten(*y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(ctx, activation, place, cur_y, cur_y);
  }
};

// Deriving the Gradient for the Backward Pass of Batch Normalization
// https://kevinzakka.github.io/2016/09/14/batch_normalization/
template <typename DeviceContext, typename T>
class InplaceABNGradKernel
    : public paddle::operators::SyncBatchNormGradKernel<DeviceContext, T>,
      public paddle::operators::BatchNormGradKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* y = ctx.Input<Tensor>("Y");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    PADDLE_ENFORCE_EQ(d_x, d_y,
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

    if (ctx.Attr<bool>("use_sync_bn")) {
      SyncBatchNormGradKernel<DeviceContext, T>::Compute(ctx);
    } else {
      BatchNormGradKernel<DeviceContext, T>::Compute(ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(inplace_abn,
                        ops::InplaceABNKernel<plat::CUDADeviceContext, float>,
                        ops::InplaceABNKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    inplace_abn_grad, ops::InplaceABNGradKernel<plat::CUDADeviceContext, float>,
    ops::InplaceABNGradKernel<plat::CUDADeviceContext, double>);
