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

#include "paddle/phi/kernels/index_get_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {

template <typename T, size_t Rank>
__global__ void index_get_cuda_kernel(const int64_t N,
                                      T* x,
                                      int64_t** indices,
                                      phi::Array<int64_t, Rank> stride,
                                      phi::Array<int64_t, Rank> shape,
                                      T* out) {
  int64_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  int64_t cur_ix = 0;

  if (idx >= N) {
    return;
  }
  int64_t offset = 0;
  for (int i = 0; i < Rank; ++i) {
    cur_ix = (int64_t(*(indices[i] + idx)));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }
  *(out + idx) = *(x + offset);
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexGetCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<const DenseTensor*>& indices_v,
                              DenseTensor* out) {
  auto* x_data = const_cast<T*>(x.data<T>());
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int64_t numel = out->numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  auto pd_indices = GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
  index_get_cuda_kernel<T, Rank>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          numel, x_data, pd_indices, stride_a, shape_a, out_data);
}

template <typename T, typename Context>
void IndexGetKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices_v,
                    DenseTensor* out) {
  const size_t total_dims = x.dims().size();
  PADDLE_ENFORCE_EQ(indices_v.size(),
                    total_dims,
                    phi::errors::InvalidArgument(
                        "The size %d of indices must be equal to the size %d "
                        "of the dimension of source tensor x.",
                        indices_v.size(),
                        total_dims));
  std::vector<const DenseTensor*> indices_v_offset(indices_v.size());

  auto res_dim = out->dims();
  auto indice_dtype = indices_v[0]->dtype();

  std::vector<DenseTensor> indices_v_tmp(
      indices_v.size(), DenseTensor(indice_dtype).Resize(res_dim));

  for (size_t i = 0; i < indices_v.size(); ++i) {
    if (res_dim == indices_v[i]->dims()) {
      indices_v_offset[i] = indices_v[i];
      continue;
    }

    ExpandKernel<int64_t, Context>(dev_ctx,
                                   *indices_v[i],
                                   IntArray(phi::vectorize<int64_t>(res_dim)),
                                   &indices_v_tmp[i]);

    indices_v_offset[i] = &indices_v_tmp[i];
  }

  switch (total_dims) {
    case 1:
      LaunchIndexGetCudaKernel<T, Context, 1>(
          dev_ctx, x, indices_v_offset, out);
      break;
    case 2:
      LaunchIndexGetCudaKernel<T, Context, 2>(
          dev_ctx, x, indices_v_offset, out);
      break;
    case 3:
      LaunchIndexGetCudaKernel<T, Context, 3>(
          dev_ctx, x, indices_v_offset, out);
      break;
    case 4:
      LaunchIndexGetCudaKernel<T, Context, 4>(
          dev_ctx, x, indices_v_offset, out);
      break;
    case 5:
      LaunchIndexGetCudaKernel<T, Context, 5>(
          dev_ctx, x, indices_v_offset, out);
      break;
    case 6:
      LaunchIndexGetCudaKernel<T, Context, 6>(
          dev_ctx, x, indices_v_offset, out);
      break;
    default:
      PADDLE_THROW(phi::errors::InvalidArgument(
          "dims of input tensor should be less than 7, But received"
          "%d",
          x.dims().size()));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(index_get,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexGetKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
