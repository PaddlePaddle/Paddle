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

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif
#include <cuda_runtime.h>
#include "paddle/fluid/operators/add_equal_dim_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void add(T* x, T* y, T* z, int N) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < N; i += stride) {
    z[i] = x[i] + y[i];
  }
}

template <typename DeviceContext, typename T>
class AddEqualDimCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "AddEqualDim_CUDA_kernel_start\n";
    // intputs / outputs
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* out = context.Output<Tensor>("Out");

    const T* h_x = x->data<T>();
    const T* h_y = y->data<T>();
    T* d_o = out->mutable_data<T>(context.GetPlace());

    int numel = x->numel();
    int nBytes = numel * sizeof(T);

    T* d_x;
    T* d_y;
    cudaMalloc(reinterpret_cast<void**>(&d_x), nBytes);
    cudaMalloc(reinterpret_cast<void**>(&d_y), nBytes);

    cudaMemcpy(reinterpret_cast<void*>(d_x), reinterpret_cast<void*>(h_x),
               nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_y), reinterpret_cast<void*>(h_y),
               nBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);

    add<<<gridSize, blockSize>>>(d_x, d_y, d_o, numel);

    cudaFree(d_x);
    cudaFree(d_y);
  }
};

template <typename DeviceContext, typename T>
class AddEqualDimCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "AddEqualDimGrad_CUDA_kernel_start\n";
    // 输入
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    // 输出
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<Tensor>(framework::GradVarName("Y"));

    const T* h_dout = dout->data<T>();
    T* d_dx = dx->mutable_data<T>(context.GetPlace());
    T* d_dy = dy->mutable_data<T>(context.GetPlace());

    int numel = dout->numel();
    int nBytes = numel * sizeof(T);

    cudaMemcpy(reinterpret_cast<void*>(d_dx), reinterpret_cast<void*>(h_dout),
               nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(reinterpret_cast<void*>(d_dy), reinterpret_cast<void*>(h_dout),
               nBytes, cudaMemcpyHostToDevice);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    addequaldim, ops::AddEqualDimCUDAKernel<plat::CUDADeviceContext, float>,
    ops::AddEqualDimCUDAKernel<plat::CUDADeviceContext, double>,
    ops::AddEqualDimCUDAKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    addequaldim_grad,
    ops::AddEqualDimCUDAGradKernel<plat::CUDADeviceContext, float>,
    ops::AddEqualDimCUDAGradKernel<plat::CUDADeviceContext, double>,
    ops::AddEqualDimCUDAGradKernel<plat::CUDADeviceContext, plat::float16>);
