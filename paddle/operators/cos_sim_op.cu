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

#define EIGEN_USE_GPU
#include "paddle/operators/cos_sim_op.h"
#include "paddle/platform/cuda_helper.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void CosSimDyKernel(const T* x_norm, const T* y_norm, const T* x,
                               const T* y, const T* z, const T* dz,
                               const size_t rows, const size_t cols, T* dy) {
  int grid_size = blockDim.x * gridDim.x;
  T y_norm_data = y_norm[0];
  for (int offset = blockIdx.x * blockDim.x + threadIdx.x; offset < rows;
       offset += grid_size) {
    T xy_norm_prod = x_norm[offset] * y_norm_data;
    T dz_data = dz[offset];
    T z_data = z[offset];
    const T* x_data = x + cols * offset;
    T reciprocal_xy_norm_prod = 1 / xy_norm_prod;

    T y_norm_square = y_norm_data * y_norm_data;
    T reciprocal_y_norm_square = 1 / y_norm_square;
    for (size_t i = 0; i < cols; ++i) {
      T dy_data = dz_data * (x_data[i] * reciprocal_xy_norm_prod -
                             z_data * y[i] * reciprocal_y_norm_square);
      platform::CudaAtomicAdd(dy + i, dy_data);
    }
  }
}

template <typename T>
struct CosSimDyFunctor<platform::CUDADeviceContext, T> {
  inline void operator()(const platform::CUDADeviceContext& ctx,
                         const T* x_norm, const T* y_norm, const T* x,
                         const T* y, const T* z, const T* dz, const size_t rows,
                         const size_t cols, T* dy) const {
    const int block_size = 512;
    dim3 threads(block_size, 1);
    dim3 grid(1, (rows + block_size - 1) / block_size);
    CosSimDyKernel<T><<<grid, threads, 0, ctx.stream()>>>(
        x_norm, y_norm, x, y, z, dz, rows, cols, dy);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    cos_sim, ops::CosSimKernel<paddle::platform::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(
    cos_sim_grad,
    ops::CosSimGradKernel<paddle::platform::CUDADeviceContext, float>);
