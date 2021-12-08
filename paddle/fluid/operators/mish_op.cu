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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mish_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeMishFw(const T* in, T* out, const int numel,
                         const float threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < numel; tid += stride) {
    T x = in[tid];
    T sp = CalcSoftplus<T>(x, threshold);
    out[tid] = x * tanh(sp);
  }
}

// expf instead of exp should be used for float type, complement
// and register float kernel separatelly
__global__ void KeMishFwFP32(const float* in, float* out, const int numel,
                             const float threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < numel; tid += stride) {
    float x = in[tid];
    float sp = CalcSoftplusFP32(x, threshold);
    out[tid] = x * tanhf(sp);
  }
}

template <typename T>
__global__ void KeMishBw(const T* in, const T* dout, T* din, const int numel,
                         const float threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < numel; tid += stride) {
    T x = in[tid];
    T sp = CalcSoftplus<T>(x, threshold);
    T tsp = tanh(sp);
    T grad_sp = -expm1(-sp);
    T grad_tsp = (static_cast<T>(1) - tsp * tsp) * grad_sp;
    din[tid] = dout[tid] * (x * grad_tsp + tsp);
  }
}

__global__ void KeMishBwFP32(const float* in, const float* dout, float* din,
                             const int numel, const float threshold) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < numel; tid += stride) {
    float x = in[tid];
    float sp = CalcSoftplusFP32(x, threshold);
    float tsp = tanhf(sp);
    float grad_sp = -expm1f(-sp);
    float grad_tsp = (static_cast<float>(1) - tsp * tsp) * grad_sp;
    din[tid] = dout[tid] * (x * grad_tsp + tsp);
  }
}

template <typename DeviceContext, typename T>
class MishCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    const float threshold = ctx.Attr<float>("threshold");

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    const int numel = x->numel();

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), numel);
    KeMishFw<T><<<config.block_per_grid, config.thread_per_block, 0,
                  ctx.cuda_device_context().stream()>>>(x_data, out_data, numel,
                                                        threshold);
  }
};

template <typename DeviceContext>
class MishFP32CUDAKernel : public framework::OpKernel<float> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    const float threshold = ctx.Attr<float>("threshold");

    const float* x_data = x->data<float>();
    float* out_data = out->mutable_data<float>(ctx.GetPlace());

    const int numel = x->numel();

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), numel);
    KeMishFwFP32<<<config.block_per_grid, config.thread_per_block, 0,
                   ctx.cuda_device_context().stream()>>>(x_data, out_data,
                                                         numel, threshold);
  }
};

template <typename DeviceContext, typename T>
class MishGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto threshold = ctx.Attr<float>("threshold");

    const T* x_data = x->data<T>();
    const T* dout_data = dout->data<T>();
    T* dx_data = dx->mutable_data<T>(ctx.GetPlace());

    const int numel = x->numel();

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), numel);
    KeMishBw<T><<<config.block_per_grid, config.thread_per_block, 0,
                  ctx.cuda_device_context().stream()>>>(
        x_data, dout_data, dx_data, numel, threshold);
  }
};

template <typename DeviceContext>
class MishGradFP32CUDAKernel : public framework::OpKernel<float> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto threshold = ctx.Attr<float>("threshold");

    const float* x_data = x->data<float>();
    const float* dout_data = dout->data<float>();
    float* dx_data = dx->mutable_data<float>(ctx.GetPlace());

    const int numel = x->numel();

    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx.cuda_device_context(), numel);
    KeMishBwFP32<<<config.block_per_grid, config.thread_per_block, 0,
                   ctx.cuda_device_context().stream()>>>(
        x_data, dout_data, dx_data, numel, threshold);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    mish, ops::MishFP32CUDAKernel<paddle::platform::CUDADeviceContext>,
    ops::MishCUDAKernel<paddle::platform::CUDADeviceContext, double>)
REGISTER_OP_CUDA_KERNEL(
    mish_grad, ops::MishGradFP32CUDAKernel<paddle::platform::CUDADeviceContext>,
    ops::MishGradCUDAKernel<paddle::platform::CUDADeviceContext, double>)
