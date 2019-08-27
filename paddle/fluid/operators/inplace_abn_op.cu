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
#include "paddle/fluid/operators/sync_batch_norm_op.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InplaceABNKernel
    : public paddle::operators::SyncBatchNormKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    int activation = *ctx.Input<Tensor>("activation")->data<int>();
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    SyncBatchNormKernel<DeviceContext, T>::Compute(ctx);

    // apply in-place activation calculate
    // apply in-place activation calculate
    auto cur_x = EigenMatrix<T>::From(*x);
    auto cur_y = EigenMatrix<T>::From(*y);

    InplaceABNActivation<DeviceContext, T> functor;
    functor.Compute(activation, place, cur_x, cur_y);
  }
};

// Deriving the Gradient for the Backward Pass of Batch Normalization
// https://kevinzakka.github.io/2016/09/14/batch_normalization/
template <typename DeviceContext, typename T>
class InplaceABNGradKernel
    : public paddle::operators::SyncBatchNormGradKernel<DeviceContext, T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Y");
    const auto* d_y = ctx.Input<Tensor>(framework::GradVarName("d_y"));
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("d_x"));
    int activation = *ctx.Input<Tensor>("activation")->data<int>();
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    // apply in-place activation calculate
    auto cur_x = EigenMatrix<T>::From(*x);
    auto cur_y = EigenMatrix<T>::From(*y);
    auto cur_dx = EigenMatrix<T>::From(*d_x);
    auto cur_dy = EigenMatrix<T>::From(*d_y);
    InplaceABNActivation<DeviceContext, T> functor;
    functor.GradCompute(activation, place, cur_x, cur_y, cur_dx, cur_dy);

    auto inp_cur_dy = EigenMatrix<T>::From(const_cast<Tensor&>(*d_y));
    inp_cur_dy.device(place) = cur_dx;
    SyncBatchNormGradKernel<DeviceContext, T>::Compute(ctx);
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
