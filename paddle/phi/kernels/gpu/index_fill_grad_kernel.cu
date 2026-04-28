// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_fill_grad_kernel.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/index_fill_util.h"

namespace phi {

// GPU kernel for index_fill backward pass.
//
// Gradient logic:
//   Forward: out[..., index[i], ...] = fill_value (a constant scalar)
//   Since the filled positions are overwritten with a constant, their
//   gradient w.r.t. x is zero. All other positions have gradient = out_grad.
//
//   So the backward is:
//     1) x_grad = copy(out_grad)
//     2) x_grad[..., index[i], ...] = 0   (this kernel does step 2)
//
//   There is no value_grad because `value` is a scalar constant, not a tensor.
//
// Uses the same three-segment decomposition as the forward kernel.
template <typename T>
__global__ void IndexFillGradCudaKernel(const int64_t* index,
                                        const int64_t index_size,
                                        const int64_t dim_size,
                                        const int64_t outer_size,
                                        const int64_t inner_size,
                                        T* x_grad) {
  int64_t idx =
      static_cast<int64_t>(threadIdx.x) +
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(blockIdx.x);
  int64_t total = index_size * outer_size * inner_size;

  if (idx >= total) {
    return;
  }

  // Same three-segment coordinate decomposition as the forward kernel.
  int64_t inner_idx = idx % inner_size;
  int64_t temp = idx / inner_size;
  int64_t index_idx = temp % index_size;
  int64_t outer_idx = temp / index_size;

  int64_t dim_idx = index[index_idx];
  if (dim_idx < 0) {
    dim_idx += dim_size;
  }

  if (dim_idx < 0 || dim_idx >= dim_size) {
    return;
  }

  int64_t offset =
      outer_idx * dim_size * inner_size + dim_idx * inner_size + inner_idx;

  // Zero out the gradient at filled positions, because forward wrote a
  // constant.
  *(x_grad + offset) = static_cast<T>(0);
}

// Host-side launch function for the backward kernel.
template <typename T, typename Context>
void LaunchIndexFillGradCudaKernel(const Context& dev_ctx,
                                   const DenseTensor& index,
                                   const DenseTensor& out_grad,
                                   const int dim,
                                   DenseTensor* x_grad) {
  // Step 1: x_grad = out_grad (full copy first, then zero out selected
  // positions)
  Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  // Cast index to int64 if needed (same logic as forward kernel).
  DenseTensor index_int64;
  const DenseTensor* ptr_index = nullptr;

  if (index.dtype() == DataType::INT32) {
    index_int64.Resize(index.dims());
    dev_ctx.template Alloc<int64_t>(&index_int64);

    int64_t index_numel = index.numel();
    auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, index_numel);

    funcs::CastToInt64Kernel<int32_t><<<config.block_per_grid,
                                        config.thread_per_block,
                                        0,
                                        dev_ctx.stream()>>>(
        index.data<int32_t>(), index_int64.data<int64_t>(), index_numel);

    ptr_index = &index_int64;
  } else if (index.dtype() == DataType::INT64) {
    ptr_index = &index;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The dtype of index must be int32 or int64, but received %s.",
        DataTypeToString(index.dtype())));
  }

  const int64_t* index_data = ptr_index->data<int64_t>();
  int64_t index_size = ptr_index->numel();

  if (index_size == 0) {
    return;
  }

  // Three-segment decomposition (same as forward).
  int64_t outer_size = 1;
  int64_t inner_size = 1;
  int64_t dim_size = out_grad_dims[dim];

  for (int i = 0; i < dim; ++i) {
    outer_size *= out_grad_dims[i];
  }
  for (int i = dim + 1; i < rank; ++i) {
    inner_size *= out_grad_dims[i];
  }

  // Step 2: launch kernel to zero out gradients at the filled positions.
  int64_t numel = outer_size * index_size * inner_size;
  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, numel);

  T* x_grad_data = x_grad->data<T>();

  IndexFillGradCudaKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, dev_ctx.stream()>>>(
          index_data,
          index_size,
          dim_size,
          outer_size,
          inner_size,
          x_grad_data);
}

// Top-level backward kernel entry: validates inputs and dispatches.
template <typename T, typename Context>
void IndexFillGradKernel(const Context& dev_ctx,
                         const DenseTensor& index,
                         const DenseTensor& out_grad,
                         int dim,
                         DenseTensor* x_grad) {
  if (out_grad.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }

  dev_ctx.template Alloc<T>(x_grad);

  auto out_grad_dims = out_grad.dims();
  const int rank = out_grad_dims.size();

  if (dim < 0) {
    dim += rank;
  }

  PADDLE_ENFORCE_GE(
      dim,
      0,
      common::errors::InvalidArgument("The dimension index should be greater "
                                      "than or equal to 0, but got %d.",
                                      dim));
  PADDLE_ENFORCE_LT(
      dim,
      rank,
      common::errors::InvalidArgument(
          "The dimension index should be less than rank %d, but got %d.",
          rank,
          dim));

  if (index.numel() == 0) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    return;
  }

  LaunchIndexFillGradCudaKernel<T, Context>(
      dev_ctx, index, out_grad, dim, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_fill_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::IndexFillGradKernel,
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
