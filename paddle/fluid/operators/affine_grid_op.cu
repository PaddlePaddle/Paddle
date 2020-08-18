/* Copyright (c) 2010 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/affine_grid_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
__global__ void LinspaceKernel(T start, T step, int64_t size, T* out) {
  CUDA_KERNEL_LOOP(index, size) { out[index] = start + step * index; }
}

template <typename T>
struct Linspace<paddle::platform::CUDADeviceContext, T> {
  void operator()(T start, T end, int count, bool align_corners,
                  framework::Tensor* numbers,
                  const framework::ExecutionContext& ctx) {
    T* number_data = numbers->mutable_data<T>({count}, ctx.GetPlace());
    T slice = (end - start) / (T)(count - 1);
    if (!align_corners) {
      slice = (end - start) / (T)count;
      start *= (T)(count - 1) / (T)count;
    }
    auto stream = ctx.cuda_device_context().stream();
    int block = 512;
    int grid = (count + block - 1) / block;
    LinspaceKernel<T><<<grid, block, 0, stream>>>(start, slice, count,
                                                  number_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    affine_grid,
    ops::AffineGridOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AffineGridOpKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    affine_grid_grad,
    ops::AffineGridGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::AffineGridGradOpKernel<paddle::platform::CUDADeviceContext, double>);
