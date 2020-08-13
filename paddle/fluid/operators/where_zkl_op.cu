/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/fluid/platform/device_context.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/where_zkl_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <class T>
__global__ void WhereZklCUDAKernel(const bool* condition, const T* x,
                                   const T* y, T* out, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    if (condition[id]) {
      out[id] = x[id];
    } else {
      out[id] = y[id];
    }

    id += blockDim.x * gridDim.x;
  }
}

template <typename T>
// class WhereZklGPUKernel<platform::CUDADeviceContext, T>
class WhereZklGPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(context.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();

    auto* condition = context.Input<Tensor>("Condition");
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* out = context.Output<Tensor>("Out");

    auto* condtion_data = condition->data<bool>();
    auto* x_data = x->data<T>();
    auto* y_data = y->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    // int n = static_cast<int>(x_data->size());
    // int n = 4;

    auto x_dims = x->dims();
    int n = static_cast<int>(framework::product(x_dims));

    int thread_per_block = 256;
    int block_per_grid = (n + thread_per_block - 1) / thread_per_block;

    WhereZklCUDAKernel<T><<<block_per_grid, thread_per_block, 0, stream>>>(
        condtion_data, x_data, y_data, out_data, n);
  }
};

template <class T>
__global__ void WhereZklCUDAGradKernel(const bool* condition, const T* out,
                                       T* dx, T* dy, int64_t N) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < N) {
    if (condition[id]) {
      dx[id] = out[id];
      dy[id] = 0;
    } else {
      dy[id] = out[id];
      dx[id] = 0;
    }

    id += blockDim.x * gridDim.x;
  }
}

template <typename T>
// class WhereZklGPUGradKernel<platform::CUDADeviceContext, T>
class WhereZklGPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(context.GetPlace()), true,
                      platform::errors::PreconditionNotMet(
                          "This kernel only runs on GPU device."));

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();

    auto* condition = context.Input<Tensor>("Condition");
    auto* out = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<Tensor>(framework::GradVarName("Y"));

    auto* condtion_data = condition->data<bool>();
    auto* out_data = out->data<T>();

    auto* dx_data = dx->mutable_data<T>(context.GetPlace());
    auto* dy_data = dy->mutable_data<T>(context.GetPlace());

    auto out_dims = out->dims();
    int n = static_cast<int>(framework::product(out_dims));

    int thread_per_block = 256;
    int block_per_grid = (n + thread_per_block - 1) / thread_per_block;

    WhereZklCUDAGradKernel<T><<<block_per_grid, thread_per_block, 0, stream>>>(
        condtion_data, out_data, dx_data, dy_data, n);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

// REGISTER_OP_CUDA_KERNEL(
//    where_zkl,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    float>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    double>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    uint8_t>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    int8_t>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    int16_t>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    int>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    int64_t>,
//    paddle::operators::WhereZklGPUKernel<paddle::platform::CUDADeviceContext,
//    plat::float16>);

REGISTER_OP_CUDA_KERNEL(where_zkl, paddle::operators::WhereZklGPUKernel<float>,
                        paddle::operators::WhereZklGPUKernel<double>,
                        paddle::operators::WhereZklGPUKernel<int>,
                        paddle::operators::WhereZklGPUKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(where_zkl_grad,
                        paddle::operators::WhereZklGPUGradKernel<float>,
                        paddle::operators::WhereZklGPUGradKernel<double>,
                        paddle::operators::WhereZklGPUGradKernel<int>,
                        paddle::operators::WhereZklGPUGradKernel<int64_t>);
