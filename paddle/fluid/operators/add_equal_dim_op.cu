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
#include "paddle/fluid/operators/add_equal_dim_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void add(const T* x, const T* y, T* z, int N) {
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

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    const T* y_ptr = y->data<T>();
    T* out_ptr = out->mutable_data<T>(context.GetPlace());

    int numel = x->numel();

    dim3 blockSize(256);
    dim3 gridSize((numel + blockSize.x - 1) / blockSize.x);
    add<<<gridSize, blockSize>>>(x_ptr, y_ptr, out_ptr, numel);
  }
};

template <typename DeviceContext, typename T>
class AddEqualDimCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    VLOG(3) << "AddEqualDimGrad_CUDA_kernel_start\n";

    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<Tensor>(framework::GradVarName("Y"));

    const T* dout_const_ptr = dout->data<T>();
    T* dout_ptr = const_cast<T*>(dout_const_ptr);

    T* dx_ptr = dx->mutable_data<T>(context.GetPlace());
    T* dy_ptr = dy->mutable_data<T>(context.GetPlace());

    int numel = dout->numel();
    int nBytes = numel * sizeof(T);

    cudaMemcpy(reinterpret_cast<void*>(dx_ptr),
               reinterpret_cast<void*>(dout_ptr), nBytes,
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(reinterpret_cast<void*>(dy_ptr),
               reinterpret_cast<void*>(dout_ptr), nBytes,
               cudaMemcpyDeviceToDevice);
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
