// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

template <typename T>
__global__ void IndexGetCudaKernel(const T* x,
                                   int64_t** indices,
                                   Array<int64_t, DDim::kMaxRank> stride,
                                   Array<int64_t, DDim::kMaxRank> shape,
                                   const int rank,
                                   const int64_t numel,
                                   T* out) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);

  if (idx >= numel) {
    return;
  }
  int64_t offset = 0;
#pragma unroll
  for (int i = 0; i < DDim::kMaxRank; ++i) {
    if (i >= rank) {
      break;
    }
    int64_t cur_ix = static_cast<int64_t>(*(indices[i] + idx));
    if (cur_ix < 0) {
      cur_ix += shape[i];
    }
    offset += stride[i] * cur_ix;
  }
  out[idx] = x[offset];
}

template <typename T, typename Context>
void LaunchIndexGetCudaKernel(const Context& dev_ctx,
                              const DenseTensor& x,
                              const std::vector<const DenseTensor*>& indices,
                              DenseTensor* out) {
  auto* x_data = x.data<T>();
  auto* out_data = dev_ctx.template Alloc<T>(out);

  auto x_dims = x.dims();
  const int rank = x_dims.size();
  auto x_stride = common::stride(x_dims);

  Array<int64_t, DDim::kMaxRank> stride_array;
  Array<int64_t, DDim::kMaxRank> shape_array;
  for (int i = 0; i < rank; ++i) {
    stride_array[i] = x_stride[i];
    shape_array[i] = x_dims[i];
  }

  const int64_t numel = indices[0]->numel();
  Allocator::AllocationPtr holder;
  auto pd_indices =
      funcs::GetDevicePointerArray<int64_t, Context>(dev_ctx, indices, &holder);

  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);
  IndexGetCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          x_data, pd_indices, stride_array, shape_array, rank, numel, out_data);
}

template <typename T, typename Context>
void IndexGetKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const std::vector<const DenseTensor*>& indices,
                    DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  std::vector<DenseTensor> temp_args;
  std::vector<const DenseTensor*> int_indices =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &temp_args);
  if (int_indices.empty()) {
    // All bool indices are all-false → output is zero-size with trailing dims
    int64_t effective_num = 0;
    for (const auto* idx : indices) {
      if (idx->dtype() == DataType::BOOL) {
        effective_num += idx->dims().size();
      } else {
        effective_num += 1;
      }
    }
    std::vector<int64_t> out_shape;
    out_shape.push_back(0);
    for (int64_t i = effective_num; i < x.dims().size(); ++i) {
      out_shape.push_back(x.dims()[i]);
    }
    out->Resize(common::make_ddim(out_shape));
    dev_ctx.template Alloc<T>(out);
    return;
  }
  auto bd_dim = funcs::BroadCastTensorsDims(int_indices);

  std::vector<int64_t> res_dim_v(vectorize(bd_dim));
  std::vector<const DenseTensor*> res_indices(x.dims().size(), nullptr);
  std::vector<DenseTensor> tmp_res_indices;
  std::vector<DenseTensor> range_tensors;

  for (int i = static_cast<int>(int_indices.size()); i < x.dims().size(); ++i) {
    range_tensors.emplace_back(funcs::GetRangeCudaTensor<int64_t, Context>(
        dev_ctx, x.dims()[i], DataType::INT64));
  }

  funcs::DealWithIndices<T, Context>(dev_ctx,
                                     x,
                                     int_indices,
                                     &res_indices,
                                     &tmp_res_indices,
                                     range_tensors,
                                     bd_dim,
                                     &res_dim_v);

  // Resize output to correct shape (may differ from infer meta when bool
  // indices are present, since the exact number of True elements is only
  // known after NonZero)
  out->Resize(common::make_ddim(res_dim_v));
  LaunchIndexGetCudaKernel<T, Context>(dev_ctx, x, res_indices, out);
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
                   int16_t,
                   uint8_t,
                   int8_t,
                   phi::float16,
                   phi::bfloat16,
                   phi::complex64,
                   phi::complex128) {}
