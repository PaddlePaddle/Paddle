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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/diag_kernel.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

// Extract the diagonal of a matrix 'dout' to a matrix 'dx'
template <typename T>
__global__ void ExtractDiagonalKernel(const T* dout,
                                      T* dx,
                                      std::ptrdiff_t start,
                                      std::ptrdiff_t dx_length,
                                      const std::ptrdiff_t sumStride,
                                      const std::ptrdiff_t xStride) {
  for (std::ptrdiff_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < dx_length;
       idx += gridDim.x * blockDim.x) {
    const std::ptrdiff_t outOffset = start + sumStride * idx;
    dx[xStride * idx] = dout[outOffset];
  }
}

// Paste a vector 'dout' to the diagonal of a matrix 'dx'
template <typename T>
__global__ void PasteDiagonalKernel(const T* dout,
                                    T* dx,
                                    std::ptrdiff_t start,
                                    std::ptrdiff_t size,
                                    const std::ptrdiff_t sumStride,
                                    const std::ptrdiff_t outStride) {
  for (std::ptrdiff_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    std::ptrdiff_t xOffset = start + sumStride * idx;
    dx[xOffset] = dout[outStride * idx];
  }
}

template <typename T, typename Context>
void DiagGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    int offset,
                    DenseTensor* x_grad) {
  T* dx_data = dev_ctx.template Alloc<T>(x_grad);
  auto* dout_data = out_grad.data<T>();
  auto dx_dims = x_grad->dims();
  auto dout_dims = out_grad.dims();

  auto GetBlockGridSize = [&dev_ctx](int64_t size) {
    const int64_t block_size =
        std::min(size, static_cast<int64_t>(dev_ctx.GetMaxThreadsPerBlock()));
    int64_t max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int64_t max_blocks =
        std::max(((max_threads - 1) / block_size + 1), static_cast<int64_t>(1));
    const int64_t grid_size =
        std::min(max_blocks, (size + block_size - 1) / block_size);
    return std::tuple<int64_t, int64_t>{block_size, grid_size};
  };

  if (dx_dims.size() <= 1) {
    auto dx_length = (dx_dims.size() == 1 ? dx_dims[0] : int64_t(1));
    auto size = (offset > 0) ? dx_length + offset : dx_length - offset;
    int dx_stride = 1;
    if (size > 0) {
      auto dout_stride_0 = phi::funcs::ComputeStride(0, dout_dims);
      auto dout_stride_1 = phi::funcs::ComputeStride(1, dout_dims);
      auto start =
          (offset >= 0 ? offset * dout_stride_1 : -offset * dout_stride_0);

      std::tuple<int64_t, int64_t> block_grid_size = GetBlockGridSize(size);
      ExtractDiagonalKernel<T>
          <<<std::get<1>(block_grid_size),
             std::get<0>(block_grid_size),
             0,
             dev_ctx.stream()>>>(dout_data,
                                 dx_data,
                                 start,
                                 dx_length,
                                 dout_stride_0 + dout_stride_1,
                                 dx_stride);
    }
  } else {
    phi::funcs::SetConstant<Context, T> set_padding_value;
    set_padding_value(dev_ctx, x_grad, static_cast<T>(0));

    int dx_stride_0 = phi::funcs::ComputeStride(0, dx_dims);
    int dx_stride_1 = phi::funcs::ComputeStride(1, dx_dims);
    int64_t size;
    if (offset > 0) {
      size = std::min(dx_dims[0], dx_dims[1] - offset);
    } else {
      size = std::min(dx_dims[0] + offset, dx_dims[1]);
    }

    if (size > 0) {
      auto start = (offset >= 0 ? offset * dx_stride_1 : -offset * dx_stride_0);
      auto dout_stride_0 = phi::funcs::ComputeStride(0, dout_dims);
      std::tuple<int64_t, int64_t> block_grid_size = GetBlockGridSize(size);
      PasteDiagonalKernel<T><<<std::get<1>(block_grid_size),
                               std::get<0>(block_grid_size),
                               0,
                               dev_ctx.stream()>>>(dout_data,
                                                   dx_data,
                                                   start,
                                                   size,
                                                   dx_stride_0 + dx_stride_1,
                                                   dout_stride_0);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(diag_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DiagGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
