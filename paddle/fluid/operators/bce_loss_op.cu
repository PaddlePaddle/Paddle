/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <algorithm>
#include "paddle/fluid/operators/bce_loss_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
struct BCELossGradFunctor {
  T one = static_cast<T>(1.0f);
  T eps = static_cast<T>(1e-12);
  __device__ __forceinline__ T operator()(const T x, const T label,
                                          const T dout) const {
    T term1 = max((one - x) * x, eps);
    return (dout * (x - label) / term1);
  }
};

template <typename T>
__global__ void GPUBCELossForward(const T* x_data, const T* label_data,
                                  T* out_data, const int in_numel) {
  CUDA_KERNEL_LOOP(i, in_numel) {
    T x = x_data[i];
    T label = label_data[i];
    T one = static_cast<T>(1.);
    T neg_100 = static_cast<T>(-100.);

    PADDLE_ENFORCE(
        (x >= static_cast<T>(0)) && (x <= one),
        "Input is expected to be within the interval [0, 1], but recieved %f.",
        x);

    T term1 = max(real_log(x), neg_100);
    T term2 = max(real_log(one - x), neg_100);

    out_data[i] = ((label - one) * term2) - (label * term1);
  }
}

template <typename DeviceContext, typename T>
class BCELossCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* out = ctx.Output<Tensor>("Out");

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto x_numel = x->numel();

    auto& dev_ctx = ctx.cuda_device_context();
    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(dev_ctx, x_numel);

    GPUBCELossForward<T><<<config.block_per_grid, config.thread_per_block, 0,
                           dev_ctx.stream()>>>(x_data, labels->data<T>(),
                                               out_data, x_numel);
  }
};

template <typename DeviceContext, typename T>
class BCELossGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    std::vector<const framework::Tensor*> ins = {x, labels, dout};
    std::vector<framework::Tensor*> outs = {dx};
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto functor = BCELossGradFunctor<T>();
    paddle::operators::LaunchSameDimsElementwiseCudaKernel<
        ElementwiseType::kTernary, T, T>(dev_ctx, ins, &outs, functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    bce_loss,
    ops::BCELossCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::BCELossCUDAKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    bce_loss_grad,
    ops::BCELossGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::BCELossGradCUDAKernel<paddle::platform::CUDADeviceContext, double>);
