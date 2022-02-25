// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/mv_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace phi {

template <typename T>
__global__ void MVGradDxCUDAKernel(
    const int m, const int n, const T *dout, const T *vec, T *dx) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < m * n; idx += blockDim.x * gridDim.x) {
    int i = idx / n;
    int j = idx % n;
    dx[idx] = dout[i] * vec[j];
  }
}

template <typename T, typename Context>
void MvGradKernel(const Context &dev_ctx,
                  const DenseTensor &x,
                  const DenseTensor &vec,
                  const DenseTensor &out_grad,
                  DenseTensor *x_grad,
                  DenseTensor *vec_grad) {
  auto dout = out_grad;
  auto dx = x_grad;
  auto dvec = vec_grad;

  auto dim_x = x.dims();
  int m = dim_x[0];
  int n = dim_x[1];

  // get data ptr
  const T *x_data = x.data<T>();
  const T *vec_data = vec.data<T>();
  const T *dout_data = dout.data<T>();

  auto blas = phi::funcs::GetBlas<Context, T>(dev_ctx);
  auto stream = dev_ctx.stream();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, m * n);

  if (dx) {
    T *dx_data = dev_ctx.template Alloc<T>(dx);

    MVGradDxCUDAKernel<
        T><<<config.block_per_grid.x, config.thread_per_block.x, 0, stream>>>(
        m, n, dout_data, vec_data, dx_data);
  }

  if (dvec) {
    T *dvec_data = dev_ctx.template Alloc<T>(dvec);

    blas.GEMV(true,
              dim_x[0],
              dim_x[1],
              static_cast<T>(1),
              x_data,
              dout_data,
              static_cast<T>(0),
              dvec_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(mv_grad, GPU, ALL_LAYOUT, phi::MvGradKernel, float, double) {
}
