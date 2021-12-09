// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/where_op.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace platform = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
__global__ void WhereCUDAKernel(const int N, const bool* cond, const T* x,
                                const T* y, T* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out[idx] = cond[idx] ? x[idx] : y[idx];
  }
}

template <typename T>
__global__ void WhereGradCUDAKernel(const int N, const T* dout,
                                    const bool* cond, T* dx, T* dy) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    if (dx != nullptr) {
      dx[idx] = cond[idx] ? dout[idx] : 0.;
    }
    if (dy != nullptr) {
      dy[idx] = cond[idx] ? 0. : dout[idx];
    }
  }
}

template <typename T>
class WhereKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");
    auto numel = condition->numel();

    // TODO(GaaoWei8): Input of where can be broadcast
    const bool* cond_data = condition->data<bool>();
    const T* x_data = X->data<T>();
    const T* y_data = Y->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    auto stream = context.cuda_device_context().stream();
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto config = GetGpuLaunchConfig1D(dev_ctx, numel);
    WhereCUDAKernel<
        T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
        numel, cond_data, x_data, y_data, out_data);
  }
};

template <typename T>
class WhereGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* condition = context.Input<framework::Tensor>("Condition");
    const bool* cond_data = condition->data<bool>();
    auto numel = condition->numel();

    auto* dout_t =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx_t = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy_t = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    auto* dout = dout_t->data<T>();
    T* dx =
        (dx_t != nullptr) ? dx_t->mutable_data<T>(context.GetPlace()) : nullptr;
    T* dy =
        (dy_t != nullptr) ? dy_t->mutable_data<T>(context.GetPlace()) : nullptr;

    auto stream = context.cuda_device_context().stream();
    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto config = GetGpuLaunchConfig1D(dev_ctx, condition->numel());
    WhereGradCUDAKernel<
        T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
        numel, dout, cond_data, dx, dy);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    where, paddle::operators::WhereKernel<platform::CUDADeviceContext, float>,
    paddle::operators::WhereKernel<platform::CUDADeviceContext, double>,
    paddle::operators::WhereKernel<platform::CUDADeviceContext, int>,
    paddle::operators::WhereKernel<platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    where_grad,
    paddle::operators::WhereGradKernel<platform::CUDADeviceContext, float>,
    paddle::operators::WhereGradKernel<platform::CUDADeviceContext, double>,
    paddle::operators::WhereGradKernel<platform::CUDADeviceContext, int>,
    paddle::operators::WhereGradKernel<platform::CUDADeviceContext, int64_t>);
