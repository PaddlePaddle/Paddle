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

#include "paddle/fluid/operators/trunc_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void Trunc(const T* x, T* out, int N) {
  CUDA_KERNEL_LOOP(index, N) { out[index] = trunc(x[index]); }
}

template <typename T>
__global__ void TruncGrad(const T* dout, T* dx, int N) {
  CUDA_KERNEL_LOOP(index, N) { dx[index] = 0.0; }
}

template <typename T>
class TruncCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    const T* x_data = x->data<T>();
    T* out_data = out->mutable_data<T>(context.GetPlace());

    int numel = x->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    Trunc<<<gridSize, blockSize>>>(x_data, out_data, numel);
  }
};

template <typename T>
class TruncCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    const T* dout_data = dout->data<T>();
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    int numel = dout->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    TruncGrad<<<gridSize, blockSize>>>(dout_data, dx_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(trunc, ops::TruncCUDAKernel<float>,
                        ops::TruncCUDAKernel<double>,
                        ops::TruncCUDAKernel<int>);

REGISTER_OP_CUDA_KERNEL(trunc_grad, ops::TruncCUDAGradKernel<float>,
                        ops::TruncCUDAGradKernel<double>,
                        ops::TruncCUDAGradKernel<int>);
