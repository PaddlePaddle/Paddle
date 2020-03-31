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

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include "paddle/fluid/operators/where_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void WhereCudaKernel(const int N, const bool* cond, const T* x,
                                const T* y, T* out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    out[idx] = cond[idx] ? x[idx] : y[idx];
  }
}

template <typename T>
__global__ void WhereGradCudaKernel(const int N, const T* out, const bool* cond,
                                    T* x, T* y) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < N; idx += blockDim.x * gridDim.x) {
    x[idx] = out[idx] * (cond[idx] ? 1. : 0.);
    y[idx] = out[idx] * (cond[idx] ? 0. : 1.);
  }
}

template <typename T>
class WhereOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use CUDAPlace.");
    auto* condition = context.Input<framework::Tensor>("Condition");
    auto* X = context.Input<framework::Tensor>("X");
    auto* Y = context.Input<framework::Tensor>("Y");
    auto* out = context.Output<framework::Tensor>("Out");

    auto x_numel = X->numel();
    auto y_numel = Y->numel();
    // TODO(GaaoWei8): Input of where can be broadcast
    PADDLE_ENFORCE_EQ(
        x_numel, y_numel,
        platform::errors::InvalidArgument(
            "X's length (%d) is not equal with Y' (%d).", x_numel, y_numel));

    const bool* cond_data = condition->data<bool>();
    const T* x_data = X->data<T>();
    const T* y_data = Y->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    auto stream = context.cuda_device_context().stream();
    WhereCudaKernel<T><<<1, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        x_numel, cond_data, x_data, y_data, out_data);
  }
};

template <typename T>
class WhereGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(context.GetPlace()),
                   "It must use CUDAPlace.");

    auto* condition = context.Input<framework::Tensor>("Condition");
    const bool* cond_data = condition->data<bool>();
    auto numel = condition->numel();

    auto* dout_t =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx_t = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy_t = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    auto dout = dout_t->data<T>();
    auto dx = dx_t->mutable_data<T>(context.GetPlace());
    auto dy = dy_t->mutable_data<T>(context.GetPlace());

    auto stream = context.cuda_device_context().stream();
    WhereGradCudaKernel<T><<<1, PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
        numel, dout, cond_data, dx, dy);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(where, paddle::operators::WhereOpCUDAKernel<float>,
                        paddle::operators::WhereOpCUDAKernel<double>,
                        paddle::operators::WhereOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(where_grad,
                        paddle::operators::WhereGradOpCUDAKernel<float>,
                        paddle::operators::WhereGradOpCUDAKernel<double>,
                        paddle::operators::WhereGradOpCUDAKernel<int>);
