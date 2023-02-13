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

#include "paddle/phi/kernels/index_put_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/array.h"
#include "paddle/phi/kernels/expand_kernel.h"

namespace phi {

// TODO(LiuYang): Here when ix is negative, we need extra error handling code
template <typename T, size_t Rank>
__global__ void index_put_cuda_kernel(const int64_t N,
                                      T* x,
                                      const T* vals,
                                      int64_t** indices,
                                      phi::Array<int64_t, Rank> stride,
                                      phi::Array<int64_t, Rank> shape,
                                      int64_t isSingleValTensor,
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

  *(x + offset) = *(vals + (idx & isSingleValTensor));
  // Note(LiuYang):Here temp test just for add backward,
  //  *(out + offset) = *(vals + (idx & isSingleValTensor));
}

template <typename T, typename Context, size_t Rank>
void LaunchIndexPutCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<const DenseTensor*>& indices_v,
                              const DenseTensor& value,
                              DenseTensor* out) {
  auto* x_data = const_cast<T*>(x.data<T>());
  auto* val_data = value.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int64_t numel = indices_v[0]->numel();
  auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  auto x_stride = phi::stride(x_dims);

  phi::Array<int64_t, Rank> stride_a;
  phi::Array<int64_t, Rank> shape_a;

  for (size_t idx = 0; idx < Rank; ++idx) {
    stride_a[idx] = x_stride[idx];
    shape_a[idx] = x_dims[idx];
  }

  // Note(LiuYang):Here should we make it inplace or not?
  phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  int64_t isSingleValTensor = (value.numel() == 1) ? 0 : INT64_MAX;

  // NOTE(LiuYang)：Here I can't make it const int64_t**
  auto pd_indices = GetDevicePointerArray<int64_t, Context>(dev_ctx, indices_v);
  index_put_cuda_kernel<T, Rank>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          numel,
          x_data,
          val_data,
          pd_indices,
          stride_a,
          shape_a,
          isSingleValTensor,
          out_data);
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
  std::vector<const DenseTensor*> indices_v_offset(indices_v.size());

  auto pre_dim = indices_v[0]->dims();
  auto tmp_dim = phi::make_ddim({0});
  auto indice_dtype = indices_v[0]->dtype();
  bool need_broadcast = false;
  for (size_t i = 1; i < indices_v.size(); ++i) {
    tmp_dim = indices_v[i]->dims();
    if (pre_dim != tmp_dim) {
      pre_dim = BroadcastTwoDims(pre_dim, tmp_dim, -1);
      need_broadcast = true;
    }
  }

  std::vector<DenseTensor> indices_v_tmp(
      indices_v.size(), DenseTensor(indice_dtype).Resize(pre_dim));

  if (need_broadcast) {
    for (size_t i = 0; i < indices_v.size(); ++i) {
      if (pre_dim == indices_v[i]->dims()) {
        indices_v_offset[i] = indices_v[i];
        continue;
      }

      ExpandKernel<int64_t, Context>(dev_ctx,
                                     *indices_v[i],
                                     IntArray(phi::vectorize<int64_t>(pre_dim)),
                                     &indices_v_tmp[i]);

      indices_v_offset[i] = &indices_v_tmp[i];
    }
  } else {
    indices_v_offset = std::move(indices_v);
  }

  switch (total_dims) {
    case 1:
      LaunchIndexPutCudaKernel<T, Context, 1>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 2:
      LaunchIndexPutCudaKernel<T, Context, 2>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 3:
      LaunchIndexPutCudaKernel<T, Context, 3>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 4:
      LaunchIndexPutCudaKernel<T, Context, 4>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 5:
      LaunchIndexPutCudaKernel<T, Context, 5>(
          dev_ctx, x, indices_v_offset, value, out);
      break;
    case 6:
      LaunchIndexPutCudaKernel<T, Context, 6>(
          dev_ctx, x, indices_v_offset, value, out);
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
                   bool,
                   phi::dtype::float16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
