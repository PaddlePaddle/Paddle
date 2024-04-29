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
#include "paddle/common/array.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void FlipCudaKernel(const T* in_data,
                               T* out_data,
                               phi::Array<int64_t, DDim::kMaxRank> shape,
                               phi::Array<int64_t, DDim::kMaxRank> stride,
                               phi::Array<int, DDim::kMaxRank> flip_dims,
                               const int rank,
                               const int64_t numel,
                               const int flip_dims_size) {
  int64_t idx =
      static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
      static_cast<int64_t>(threadIdx.x);
  if (idx >= numel) {
    return;
  }

  int64_t cur_indices = idx;
  int64_t rem = 0;
  int64_t dst_offset = 0;

#pragma unroll
  for (int i = 0; i < DDim::kMaxRank; ++i) {
    if (i >= rank) {
      break;
    }
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

template <typename T, typename Context>
void FlipKernel(const Context& dev_ctx,
                const DenseTensor& x,
                const std::vector<int>& axis,
                DenseTensor* out) {
  auto* in_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int rank = x_dims.size();
  const int64_t numel = x.numel();

  size_t flip_dims_size = axis.size();
  auto x_stride = common::stride(x_dims);

  phi::Array<int64_t, DDim::kMaxRank> stride_array;
  phi::Array<int64_t, DDim::kMaxRank> shape_array;
  phi::Array<int, DDim::kMaxRank> flip_dims_array;

  for (int i = 0; i < rank; ++i) {
    stride_array[i] = x_stride[i];
    shape_array[i] = x_dims[i];
    if (i < flip_dims_size) {
      flip_dims_array[i] = axis[i] < 0 ? axis[i] + rank : axis[i];
    } else {
      flip_dims_array[i] = 0;
    }
  }

  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  FlipCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          in_data,
          out_data,
          shape_array,
          stride_array,
          flip_dims_array,
          rank,
          numel,
          flip_dims_size);
}

}  // namespace phi

PD_REGISTER_KERNEL(flip,
                   GPU,
                   ALL_LAYOUT,
                   phi::FlipKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
