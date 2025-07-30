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

#include "paddle/phi/kernels/diag_kernel.h"

#include <algorithm>
#include <tuple>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/diag_functor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

// Extract the diagonal of a matrix 'x' to a vector 'out'.
template <typename T>
__global__ void ExtractDiagonalKernel(T* out,
                                      const T* x,
                                      int64_t start,
                                      int64_t size,
                                      const int64_t sumStride,
                                      const int64_t outStride) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    const int64_t xOffset = start + sumStride * idx;
    out[outStride * idx] = x[xOffset];
  }
}

// Paste a vector 'x' to the diagonal of a matrix 'out'
template <typename T>
__global__ void PasteDiagonalKernel(T* out,
                                    const T* x,
                                    int64_t start,
                                    int64_t x_length,
                                    const int64_t sumStride,
                                    const int64_t xStride) {
  for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < x_length;
       idx += gridDim.x * blockDim.x) {
    const int64_t outOffset = start + sumStride * idx;
    out[outOffset] = x[xStride * idx];
  }
}

template <typename T, typename Context>
void DiagKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int offset,
                float padding_value,
                DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto x_dims = x.dims();
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (out && out->numel() == 0) return;
  auto out_dims = out->dims();

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

  if (x_dims.size() <= 1) {
    phi::funcs::SetConstant<Context, T> set_padding_value;
    set_padding_value(dev_ctx, out, static_cast<T>(padding_value));

    int64_t x_length = (x_dims.size() == 1ULL ? x_dims[0] : int64_t(1));
    int64_t size = (offset > 0) ? x_length + offset : x_length - offset;
    const int64_t x_stride = 1;
    if (size > 0) {
      const int64_t out_stride_0 = phi::funcs::ComputeStride(0, out_dims);
      const int64_t out_stride_1 = phi::funcs::ComputeStride(1, out_dims);
      int64_t start =
          (offset >= 0 ? offset * out_stride_1 : -offset * out_stride_0);

      std::tuple<int64_t, int64_t> block_grid_size = GetBlockGridSize(size);

      PasteDiagonalKernel<T><<<std::get<1>(block_grid_size),
                               std::get<0>(block_grid_size),
                               0,
                               dev_ctx.stream()>>>(out_data,
                                                   x_data,
                                                   start,
                                                   x_length,
                                                   out_stride_0 + out_stride_1,
                                                   x_stride);
    }
  } else {
    const int64_t x_stride_0 = phi::funcs::ComputeStride(0, x_dims);
    const int64_t x_stride_1 = phi::funcs::ComputeStride(1, x_dims);

    int64_t size;
    if (offset > 0) {
      size = std::min(x_dims[0], x_dims[1] - offset);
    } else {
      size = std::min(x_dims[0] + offset, x_dims[1]);
    }

    if (size > 0) {
      int64_t start =
          (offset >= 0 ? offset * x_stride_1 : -offset * x_stride_0);
      const int64_t out_stride_0 = phi::funcs::ComputeStride(0, out_dims);

      std::tuple<int64_t, int64_t> block_grid_size = GetBlockGridSize(size);
      ExtractDiagonalKernel<T><<<std::get<1>(block_grid_size),
                                 std::get<0>(block_grid_size),
                                 0,
                                 dev_ctx.stream()>>>(
          out_data, x_data, start, size, x_stride_0 + x_stride_1, out_stride_0);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(diag,
                   GPU,
                   ALL_LAYOUT,
                   phi::DiagKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
