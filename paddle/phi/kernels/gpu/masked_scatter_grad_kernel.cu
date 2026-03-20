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

#include "paddle/phi/kernels/masked_scatter_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/cub.h"

namespace phi {

template <typename T>
__global__ void MaskedScatterGradXKernel(const T* out_grad_data,
                                         const bool* mask_data,
                                         const int64_t total,
                                         T* x_grad_data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  x_grad_data[idx] = mask_data[idx] ? static_cast<T>(0) : out_grad_data[idx];
}

template <typename T>
__global__ void MaskedScatterGradValueKernel(const T* out_grad_data,
                                             const bool* mask_data,
                                             const int64_t* prefix_sum_data,
                                             const int64_t total,
                                             const int64_t value_numel,
                                             T* value_grad_data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (mask_data[idx]) {
    int64_t value_idx = prefix_sum_data[idx];
    if (value_idx < value_numel) {
      value_grad_data[value_idx] = out_grad_data[idx];
    }
  }
}

template <typename T, typename Context>
void MaskedScatterGradKernel(const Context& dev_ctx,
                             const DenseTensor& mask,
                             const DenseTensor& value,
                             const DenseTensor& out_grad,
                             DenseTensor* x_grad,
                             DenseTensor* value_grad) {
  auto out_grad_dims = out_grad.dims();
  auto mask_dims = mask.dims();
  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(out_grad_dims, mask_dims, -1));
  DDim expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  if (mask_dims != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  int64_t total = out_grad.numel();
  auto stream = dev_ctx.stream();
  auto* mask_data = mask_expand.data<bool>();

  // Compute x_grad
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);
    MaskedScatterGradXKernel<T>
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            out_grad.data<T>(), mask_data, total, x_grad->data<T>());
  }

  // Compute value_grad
  if (value_grad) {
    int64_t value_numel = value_grad->numel();
    Full<T, Context>(
        dev_ctx, value_grad->dims(), static_cast<T>(0), value_grad);

    // Compute prefix sum of mask for scatter index using CUB directly
    // on bool mask (avoids extra bool->int64 transform kernel)
    DenseTensor prefix_sum;
    prefix_sum.Resize(mask_expand.dims());
    dev_ctx.template Alloc<int64_t>(&prefix_sum);
    auto* prefix_sum_data = prefix_sum.data<int64_t>();

    {
      void* temp_storage = nullptr;
      size_t temp_storage_bytes = 0;
      phi::Allocator::AllocationPtr allocation;
      for (int i = 0; i < 2; ++i) {
        PADDLE_ENFORCE_GPU_SUCCESS(
            cub::DeviceScan::ExclusiveSum(temp_storage,
                                          temp_storage_bytes,
                                          mask_data,
                                          prefix_sum_data,
                                          static_cast<int>(total),
                                          stream));
        if (i == 0 && temp_storage_bytes > 0) {
          allocation =
              phi::memory_utils::Alloc(dev_ctx.GetPlace(), temp_storage_bytes);
          temp_storage = allocation->ptr();
        }
      }
    }

    auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);
    MaskedScatterGradValueKernel<T>
        <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
            out_grad.data<T>(),
            mask_data,
            prefix_sum_data,
            total,
            value_numel,
            value_grad->data<T>());
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedScatterGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(0).SetDataType(phi::DataType::BOOL);
}
