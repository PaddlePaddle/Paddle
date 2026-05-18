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

#include "paddle/phi/kernels/index_get_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/backends/gpu/gpu_primitives.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"

namespace phi {

template <typename T>
__global__ void IndexGetGradCudaKernel(const T* out_grad,
                                       int64_t** indices,
                                       Array<int64_t, DDim::kMaxRank> stride,
                                       Array<int64_t, DDim::kMaxRank> shape,
                                       const int rank,
                                       const int64_t numel,
                                       T* x_grad) {
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
  CudaAtomicAdd(x_grad + offset, out_grad[idx]);
}

template <typename T, typename Context>
void LaunchIndexGetGradCudaKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    DenseTensor* x_grad) {
  auto* out_grad_data = out_grad.data<T>();
  auto* x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  auto x_dims = x_grad->dims();
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
  IndexGetGradCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          out_grad_data,
          pd_indices,
          stride_array,
          shape_array,
          rank,
          numel,
          x_grad_data);
}

template <typename T, typename Context>
void IndexGetGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const std::vector<const DenseTensor*>& indices,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    cudaMemsetAsync(
        x_grad->data<T>(), 0, x_grad->numel() * sizeof(T), dev_ctx.stream());
    return;
  }

  PADDLE_ENFORCE_EQ(
      indices.empty(),
      false,
      common::errors::InvalidArgument("Indices cannot be empty."));

  std::vector<DenseTensor> temp_args;
  std::vector<const DenseTensor*> int_indices =
      funcs::DealWithBoolIndices<T, Context>(dev_ctx, indices, &temp_args);
  if (int_indices.empty()) {
    dev_ctx.template Alloc<T>(x_grad);
    cudaMemsetAsync(
        x_grad->data<T>(), 0, x_grad->numel() * sizeof(T), dev_ctx.stream());
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

  dev_ctx.template Alloc<T>(x_grad);
  cudaMemsetAsync(
      x_grad->data<T>(), 0, x_grad->numel() * sizeof(T), dev_ctx.stream());

  LaunchIndexGetGradCudaKernel<T, Context>(
      dev_ctx, res_indices, out_grad, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_get_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexGetGradKernel,
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
