/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/op_registry.h"
#include "paddle/platform/assert.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__host__ __device__ T clipping_log(const T x) {
  PADDLE_ASSERT(std::is_floating_point<T>::value);
  const T kApproInf = 1e20;
  T v = log(x);
  if (v == INFINITY) {
    return kApproInf;
  }
  if (v == -INFINITY) {
    return -kApproInf;
  }
  return v;
}

template <typename T>
__global__ void CrossEntropyKernel(T* Y, const T* X, const int* label,
                                   const int N, const int D) {
  // TOOD(qingqing) define CUDA_1D_KERNEL_LOOP macro in a common file.
  // CUDA_1D_KERNEL_LOOP(i, N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    PADDLE_ASSERT(label[i] >= 0 && label[i] < D);
    Y[i] = -clipping_log(X[i * D + label[i]]);
  }
}

// TODO(qingqing): make zero setting an common function.
template <typename T>
__global__ void zero(T* X, const int N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    X[i] = 0.0;
  }
}

template <typename T>
__global__ void CrossEntropyGradientKernel(T* dX, const T* dY, const T* X,
                                           const int* label, const int N,
                                           const int D) {
  // TOOD(qingqing) define CUDA_1D_KERNEL_LOOP macro in a common file.
  // CUDA_1D_KERNEL_LOOP(i, N) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    int idx = i * D + label[i];
    dX[idx] = -dY[i] / X[idx];
  }
}

template <typename T>
class OnehotCrossEntropyOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");

    auto X = ctx.Input<Tensor>("X");
    const T* Xdata = X->data<T>();
    const int* label_data = ctx.Input<Tensor>("label")->data<int>();
    auto Y = ctx.Output<Tensor>("Y");
    Y->mutable_data<T>(ctx.GetPlace());
    T* Ydata = Y->data<T>();

    int N = X->dims()[0];
    int D = X->dims()[1];
    int block = 512;
    int grid = (N + block - 1) / block;
    // TODO(qingqing) launch kernel on specified stream
    // base on ExecutionContext.
    CrossEntropyKernel<T><<<grid, block>>>(Ydata, Xdata, label_data, N, D);
  }
};

template <typename T>
class OnehotCrossEntropyGradientOpCUDAKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "It must use GPUPlace.");

    auto X = ctx.Input<Tensor>("X");
    auto dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto dY = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto label = ctx.Input<Tensor>("label");

    auto* dXdata = dX->template mutable_data<T>(ctx.GetPlace());
    auto* dYdata = dY->template data<T>();
    auto* Xdata = X->template data<T>();
    auto* label_data = label->data<int>();

    int N = X->dims()[0];
    int D = X->dims()[1];
    int block = 512;
    int grid = (N * D + block - 1) / block;
    zero<T><<<grid, block>>>(dXdata, N * D);

    grid = (N + block - 1) / block;
    // TODO(qingqing): launch kernel on specified stream
    // base on ExecutionContext.
    CrossEntropyGradientKernel<T><<<grid, block>>>(dXdata, dYdata, Xdata,
                                                   label_data, N, D);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_GPU_KERNEL(onehot_cross_entropy,
                       ops::OnehotCrossEntropyOpCUDAKernel<float>);
REGISTER_OP_GPU_KERNEL(onehot_cross_entropy_grad,
                       ops::OnehotCrossEntropyGradientOpCUDAKernel<float>);
