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
#include "cub/cub.cuh"
#include "paddle/fluid/operators/bce_loss_op.h"
#include "paddle/fluid/operators/math.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_launch_config.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

template <typename T>
__global__ void GPUBCELossForward(const T* x_data, const T* label_data,
                                  T* out_data, const int in_numel) {
  CUDA_1D_KERNEL_LOOP(i, in_numel) {
    T x = x_data[i];
    T label = label_data[i];
    T one = static_cast<T>(1.);
    T neg_100 = static_cast<T>(-100.);

    T term1 = max(real_log(x), neg_100);
    T term2 = max(real_log(one - x), neg_100);

    out_data[i] = ((label - one) * term2) - (label * term1);
  }
}

template <typename T>
__global__ void GPUBCELossBackward(const T* x_data, const T* label_data,
                                   const T* dout_data, T* dx_data,
                                   const int in_numel) {
  CUDA_1D_KERNEL_LOOP(i, in_numel) {
    T x = x_data[i];
    T label = label_data[i];
    T dout = dout_data[i];
    T one = static_cast<T>(1.);
    T eps = static_cast<T>(1e-12);

    T term1 = max((one - x) * x, eps);

    dx_data[i] = dout * (x - label) / term1;
  }
}

template <typename DeviceContext, typename T>
class BCELossCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* labels = ctx.Input<Tensor>("Label");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_data = x->data<T>();
    auto out_data = out->mutable_data<T>(ctx.GetPlace());
    int x_numel = x->numel();
    platform::GpuLaunchConfig config =
        platform::getGpuLaunchConfig(x_numel, ctx);

    Tensor x_cpu;
    framework::TensorCopy(*x, platform::CPUPlace(), &x_cpu);
    T* x_cpu_data = x_cpu.data<T>();

    for (int i = 0; i < x_numel; ++i) {
      PADDLE_ENFORCE_GE(
          x_cpu_data[i], static_cast<T>(0),
          platform::errors::InvalidArgument(
              "Illegal input, input must be greater than  or equal to 0"));
      PADDLE_ENFORCE_LE(
          x_cpu_data[i], static_cast<T>(1),
          platform::errors::InvalidArgument(
              "Illegal input, input must be less than or equal to 1"));
    }

    auto& dev_ctx = ctx.cuda_device_context();

    GPUBCELossForward<
        T><<<config.blocks, config.threads, 0, dev_ctx.stream()>>>(
        x_data, labels->data<T>(), out_data, x_numel);
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
    auto dx_data = dx->mutable_data<T>(ctx.GetPlace());

    int x_numel = x->numel();
    platform::GpuLaunchConfig config =
        platform::getGpuLaunchConfig(x_numel, ctx);
    auto& dev_ctx = ctx.cuda_device_context();

    GPUBCELossBackward<
        T><<<config.blocks, config.threads, 0, dev_ctx.stream()>>>(
        x->data<T>(), labels->data<T>(), dout->data<T>(), dx_data, x_numel);
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
