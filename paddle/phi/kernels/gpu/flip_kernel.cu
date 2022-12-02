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

#include "paddle/phi/kernels/flip_kernel.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"

namespace phi {

template <typename T, size_t Rank>
__global__ void flip_cuda_kernel(const int64_t N,
                                 const T* in_data,
                                 T* out_data,
                                 phi::Array<int64_t, Rank> shape,
                                 phi::Array<int64_t, Rank> stride,
                                 phi::Array<int, Rank> flip_dims,
                                 int flip_dims_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int cur_indices = idx, rem = 0, dst_offset = 0;
  for (int i = 0; i < Rank; ++i) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / stride[i];
    rem = temp - cur_indices * stride[i];
    // flip the indices if it is in flip_dims
    for (int j = 0; j < flip_dims_size; ++j) {
      if (i == flip_dims[j]) {
        cur_indices = shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * stride[i];
    cur_indices = rem;
  }
  out_data[idx] = in_data[dst_offset];
}

template <typename T, typename Context, size_t N>
void LaunchFlipCudaKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const std::vector<int>& axis,
                          DenseTensor* out) {
  auto* in_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int total_dims = x_dims.size();
  const int64_t numel = x.numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, N> stride_a;
  phi::Array<int64_t, N> shape_a;
  phi::Array<int, N> flip_dims_a;
  size_t flip_dims_size = axis.size();

  for (size_t idx = 0; idx < N; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
    flip_dims_a[idx] = idx < flip_dims_size ? axis[idx] : 0;
  }

  for (size_t i = 0; i < flip_dims_a.size(); ++i) {
    if (flip_dims_a[i] < 0) {
      flip_dims_a[i] += total_dims;
    }
  }
  flip_cuda_kernel<T, N>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          numel,
          in_data,
          out_data,
          shape_a,
          stride_a,
          flip_dims_a,
          flip_dims_size);
}

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int>& axis,
                DenseTensor* out) {
  const size_t total_dims = x.dims().size();
  switch (total_dims) {
    case 1:
      LaunchFlipCudaKernel<T, Context, 1>(dev_ctx, x, axis, out);
      break;
    case 2:
      LaunchFlipCudaKernel<T, Context, 2>(dev_ctx, x, axis, out);
      break;
    case 3:
      LaunchFlipCudaKernel<T, Context, 3>(dev_ctx, x, axis, out);
      break;
    case 4:
      LaunchFlipCudaKernel<T, Context, 4>(dev_ctx, x, axis, out);
      break;
    case 5:
      LaunchFlipCudaKernel<T, Context, 5>(dev_ctx, x, axis, out);
      break;
    case 6:
      LaunchFlipCudaKernel<T, Context, 6>(dev_ctx, x, axis, out);
      break;
    case 7:
      LaunchFlipCudaKernel<T, Context, 7>(dev_ctx, x, axis, out);
      break;
    case 8:
      LaunchFlipCudaKernel<T, Context, 8>(dev_ctx, x, axis, out);
      break;
    case 9:
      LaunchFlipCudaKernel<T, Context, 9>(dev_ctx, x, axis, out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 10, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(flip,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlipKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
