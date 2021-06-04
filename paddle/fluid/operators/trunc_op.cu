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

#include "paddle/fluid/operators/trunc_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void trunc(const T* x, T* out, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    out[i] = x[i] - fmod(x[i], 1.0);
  }
}

template <typename T>
__global__ void truncGrad(const T* dout, T* dx, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    dx[i] = 0.0;
  }
}

template <typename DeviceContext, typename T>
class TruncCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "Trunc_CUDA_kernel_start\n";

    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    int numel = x->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    trunc<<<gridSize, blockSize>>>(x_data, out_data, numel);
  }
};

template <typename DeviceContext, typename T>
class TruncCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "Trunc_CUDA_Grad_kernel_start\n";

    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    const T* dout_data = dout->data<T>();
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int numel = dout->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    truncGrad<<<gridSize, blockSize>>>(dout_data, dx_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(trunc,
                        ops::TruncCUDAKernel<plat::CUDADeviceContext, float>,
                        ops::TruncCUDAKernel<plat::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    trunc_grad, ops::TruncCUDAGradKernel<plat::CUDADeviceContext, float>,
    ops::TruncCUDAGradKernel<plat::CUDADeviceContext, double>);
