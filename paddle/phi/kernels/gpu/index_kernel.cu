// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_kernel.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/stack_kernel.h"

namespace phi {

template <typename T, size_t Rank>
__global__ void index_put_cuda_kernel(const int64_t N,
                                      T* x,
                                      const int64_t* indices,
                                      const T* vals,
                                      phi::Array<int64_t, Rank> stride,
                                      T* out) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= N) {
    return;
  }

  const int64_t base = idx * Rank;
  int64_t offset = 0;

#pragma unroll
  for (int j = 0; j < Rank; ++j) {
    offset += (stride[j] * (*(indices + base + j)));
  }
  *(x + offset) = *(vals + idx);
}

template <typename T, typename Context, size_t N>
void LaunchIndexPutCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& indices,
                              const DenseTensor& value,
                              DenseTensor* out) {
  auto* x_data = const_cast<T*>(x.data<T>());
  auto* val_data = value.data<T>();
  auto* indices_data = indices.data<int64_t>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int64_t numel = value.numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, N> stride_a;

  for (size_t idx = 0; idx < N; ++idx) {
    stride_a[idx] = x_stride[idx];
  }

  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  index_put_cuda_kernel<T, N>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          numel, x_data, indices_data, val_data, stride_a, out_data);
}

template <typename T, typename Context>
void IndexPutKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    const DenseTensor& value,
                    DenseTensor* out) {
  const size_t total_dims = x.dims().size();
  PADDLE_ENFORCE_EQ(indices_v.size(),
                    total_dims,
                    phi::errors::InvalidArgument(
                        "The size %d of indices must be equal to the size %d "
                        "of the dimension of source tensor x.",
                        indices_v.size(),
                        total_dims));

  auto indices = DenseTensor(indices_v[0]->dtype());
  indices.Resize(phi::make_dim(indices_v[0]->numel(), total_dims));
  StackKernel<int64_t, Context>(dev_ctx, indices_v, 1, &indices);

  switch (total_dims) {
    case 1:
      LaunchIndexPutCudaKernel<T, Context, 1>(dev_ctx, x, indices, value, out);
      break;
    case 2:
      LaunchIndexPutCudaKernel<T, Context, 2>(dev_ctx, x, indices, value, out);
      break;
    case 3:
      LaunchIndexPutCudaKernel<T, Context, 3>(dev_ctx, x, indices, value, out);
      break;
    case 4:
      LaunchIndexPutCudaKernel<T, Context, 4>(dev_ctx, x, indices, value, out);
      break;
    case 5:
      LaunchIndexPutCudaKernel<T, Context, 5>(dev_ctx, x, indices, value, out);
      break;
    case 6:
      LaunchIndexPutCudaKernel<T, Context, 6>(dev_ctx, x, indices, value, out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 7, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_put,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexPutKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
