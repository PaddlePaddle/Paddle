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
#include "paddle/phi/core/utils/array.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T,size_t Rank>
__global__ void flip_cuda_kernel(const int N,
                                 const T* in_data,
                                 T* out_data,
                                 phi::Array<int64_t,Rank> x_shape,
                                 phi::Array<int64_t,Rank> x_stride,
                                 phi::Array<int,Rank> flip_dims,
                                 int flip_dims_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) {
    return;
  }

  int cur_indices = idx, rem = 0, dst_offset = 0;
  for (int i = 0; i < Rank; ++i) {
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

#define CALL_FLIP_CUDA_KERNEL(N)                                            \
  case N: {                                                                 \
    phi::Array<int64_t, N> _stride;                                         \
    phi::Array<int64_t, N> _shape;                                          \
    phi::Array<int, N> _flip_dims;                                          \
    for (size_t idx = 0; idx < N; ++idx) {                                  \
      _stride[idx] = x_stride_v[idx];                                       \
      _shape[idx] = x_dims_v[idx];                                          \                                   
      _flip_dims[idx] = idx<flip_dims_size?flip_dims[idx]:0;                \
    }                                                                       \
    flip_cuda_kernel<T, N>                                                  \
      <<<dim_grid, dim_block, 0, dev_ctx.stream()>>>(numel,                 \
                                                     in_data,               \
                                                     out_data,              \
                                                     _shape,                \
                                                     _stride,               \
                                                     _flip_dims,            \
                                                     flip_dims_size);       \
    break;                                                                  \
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
  const int numel = x.numel();

  int block_size = 512;
  dim3 dim_block(block_size);
  dim3 dim_grid((numel + block_size - 1) / block_size);

  for (size_t i = 0; i < flip_dims.size(); ++i) {
    if (flip_dims[i] < 0) {
      flip_dims[i] += total_dims;
    }
  }

  auto x_stride = phi::stride(x_dims);
  std::vector<int64_t> x_dims_v = phi::vectorize(x_dims);
  std::vector<int64_t> x_stride_v = phi::vectorize(x_stride);

   switch (total_dims) {
    CALL_FLIP_CUDA_KERNEL(1);
    CALL_FLIP_CUDA_KERNEL(2);
    CALL_FLIP_CUDA_KERNEL(3);
    CALL_FLIP_CUDA_KERNEL(4);
    CALL_FLIP_CUDA_KERNEL(5);
    CALL_FLIP_CUDA_KERNEL(6);
    CALL_FLIP_CUDA_KERNEL(7);
    CALL_FLIP_CUDA_KERNEL(8);
    CALL_FLIP_CUDA_KERNEL(9);
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 10, But received"
          "%d",
          x_dims.size()));
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
                
