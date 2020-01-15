// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void KeRelu2(const T* x, const int num, T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    y[i] = max(x[i], static_cast<T>(0.));
  }
}

template <typename DeviceContext, typename T>
class Relu2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* in_t = ctx.Input<Tensor>("X");
    auto* out_t = ctx.Output<Tensor>("Y");
    auto x = in_t->data<T>();
    auto y = out_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int num = in_t->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu2<T><<<grid, block, 0, dev_ctx.stream()>>>(x, num, y);
  }
};

template <typename T>
__global__ void KeRelu2Grad(const T* y, const T* dy, const int num, T* dx) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = gid; i < num; i += blockDim.x * gridDim.x) {
    dx[i] = dy[i] * (y[i] > 0 ? 1. : 0.);
  }
}

template <typename DeviceContext, typename T>
class Relu2GradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* dy_t = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* y_t = ctx.Input<Tensor>("Y");
    auto* dx_t = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dy = dy_t->data<T>();
    auto y = y_t->data<T>();
    auto dx = dx_t->mutable_data<T>(ctx.GetPlace());

    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    int num = dy_t->numel();
    int block = 512;
    int grid = (num + block - 1) / block;
    KeRelu2Grad<T><<<grid, block, 0, dev_ctx.stream()>>>(y, dy, num, dx);
  }
};

}  // namespace operators
}  // namespace paddle

using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(relu2,
                        paddle::operators::Relu2CUDAKernel<CUDA, float>,
                        paddle::operators::Relu2CUDAKernel<CUDA, double>);

REGISTER_OP_CUDA_KERNEL(relu2_grad,
                        paddle::operators::Relu2GradCUDAKernel<CUDA, float>,
                        paddle::operators::Relu2GradCUDAKernel<CUDA, double>);
