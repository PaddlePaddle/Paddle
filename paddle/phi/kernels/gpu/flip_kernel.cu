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
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void flip_cuda_kernel(const int N,
                                 const T* in_data,
                                 T* out_data,
                                 int64_t* x_shape,
                                 int64_t* x_stride,
                                 int* flip_dims,
                                 int flip_dims_size,
                                 int total_dims) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int cur_indices = idx, rem = 0, dst_offset = 0;
  for (int i = 0; i < total_dims; ++i) {
    int64_t temp = cur_indices;
    cur_indices = cur_indices / x_stride[i];
    rem = temp - cur_indices * x_stride[i];
    // flip the indices if it is in flip_dims
    for (int j = 0; j < flip_dims_size; ++j) {
      if (i == flip_dims[j]) {
        cur_indices = x_shape[i] - 1 - cur_indices;
      }
    }
    dst_offset += cur_indices * x_stride[i];
    cur_indices = rem;
  }
  out_data[idx] = in_data[dst_offset];
}

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int>& axis,
                DenseTensor* out) {
  const auto gplace = dev_ctx.GetPlace();
  auto cplace = phi::CPUPlace();
  std::vector<int> flip_dims = axis;

  auto* in_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  const int flip_dims_size = static_cast<int>(flip_dims.size());
  auto x_dims = x.dims();
  const int total_dims = x_dims.size();
  const int N = x.numel();

  int block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((N + block_size - 1) / block_size);

  for (size_t i = 0; i < flip_dims.size(); ++i) {
    if (flip_dims[i] < 0) {
      flip_dims[i] += total_dims;
    }
  }

  auto x_stride = phi::stride(x_dims);
  std::vector<int64_t> x_dims_v = phi::vectorize(x_dims);
  std::vector<int64_t> x_stride_v = phi::vectorize(x_stride);

  int bytes = total_dims * sizeof(int64_t);
  auto x_strides_array_tmp = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* x_strides_array_gpu =
      reinterpret_cast<int64_t*>(x_strides_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       x_strides_array_gpu,
                       cplace,
                       x_stride_v.data(),
                       bytes,
                       dev_ctx.stream());

  auto x_shape_array_tmp = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int64_t* x_shape_array_gpu =
      reinterpret_cast<int64_t*>(x_shape_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       x_shape_array_gpu,
                       cplace,
                       x_dims_v.data(),
                       bytes,
                       dev_ctx.stream());

  bytes = flip_dims_size * sizeof(int);
  auto flip_dims_array_tmp = paddle::memory::Alloc(
      dev_ctx.GetPlace(),
      bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  int* flip_dims_array_gpu = reinterpret_cast<int*>(flip_dims_array_tmp->ptr());
  paddle::memory::Copy(gplace,
                       flip_dims_array_gpu,
                       cplace,
                       flip_dims.data(),
                       bytes,
                       dev_ctx.stream());

  flip_cuda_kernel<T>
      <<<dim_grid, dim_block, 0, dev_ctx.stream()>>>(N,
                                                     in_data,
                                                     out_data,
                                                     x_shape_array_gpu,
                                                     x_strides_array_gpu,
                                                     flip_dims_array_gpu,
                                                     flip_dims_size,
                                                     total_dims);
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
