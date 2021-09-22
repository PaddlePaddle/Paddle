/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/determinant_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;
using Tensor = framework::Tensor;

template <typename T>
__global__ void DeterminantGrad(const size_t numel, T* out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < numel) {
    out[tid] = static_cast<T>(1);
  }
}

template <typename T>
class DeterminantGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    const T* dout_data = dout->data<T>();
    auto dout_dim = vectorize(dout->dims());

    auto* dx = context.Output<Tensor>(framework::GradVarName("Input"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dx->numel();
    for (int64_t idx = 0; idx < numel; idx++) {
      dx_data[idx] = static_cast<T>(1);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    determinant, ops::DeterminantKernel<plat::CUDADeviceContext, float>,
    ops::DeterminantKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    determinant_grad,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, float>,
    ops::DeterminantGradKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    slogdeterminant, ops::SlogDeterminantKernel<plat::CUDADeviceContext, float>,
    ops::SlogDeterminantKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    slogdeterminant_grad,
    ops::SlogDeterminantGradKernel<plat::CUDADeviceContext, float>,
    ops::SlogDeterminantGradKernel<plat::CUDADeviceContext, double>);
