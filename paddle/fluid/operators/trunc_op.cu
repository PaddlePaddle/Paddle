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
class truncFunctor {
 public:
  __device__ truncFunctor(const T x) : _x(x) {}
  __device__ T operator()() { return trunc(_x); }
  const T _x;
};

template <>
class truncFunctor<int> {
 public:
  __device__ truncFunctor(const int x) : _x(x) {}
  __device__ int operator()() { return _x; }
  const int _x;
};

template <>
class truncFunctor<int64_t> {
 public:
  __device__ truncFunctor(const int64_t x) : _x(x) {}
  __device__ int64_t operator()() { return _x; }
  const int64_t _x;
};

template <typename T>
__global__ void Trunc(const T* x, T* out, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) {
    truncFunctor<T> functor(x[index]);
    out[index] = functor();
  }
}

template <typename T>
__global__ void TruncGrad(T* dx, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) { dx[index] = 0.0; }
}

template <typename T>
class TruncCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    int64_t numel = x->numel();

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

    const auto* dout_data = dout->data<T>();
    auto* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dout->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    TruncGrad<<<gridSize, blockSize>>>(dx_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(trunc, ops::TruncCUDAKernel<float>,
                        ops::TruncCUDAKernel<double>, ops::TruncCUDAKernel<int>,
                        ops::TruncCUDAKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(trunc_grad, ops::TruncCUDAGradKernel<float>,
                        ops::TruncCUDAGradKernel<double>,
                        ops::TruncCUDAGradKernel<int>,
                        ops::TruncCUDAGradKernel<int64_t>);
