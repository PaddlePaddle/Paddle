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

#include "paddle/phi/kernels/masked_scatter_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/cub.h"

namespace phi {

__global__ void BoolToInt64Kernel(const bool* in, int64_t* out, int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = static_cast<int64_t>(in[idx]);
  }
}

// Mask exclusive sum: converts bool mask to int64, then runs CUB ExclusiveSum.
// hipcub on ROCm/DCU does not reliably handle mismatched input (bool*) and
// output (int64_t*) types in ExclusiveSum, so we cast explicitly.
static void MaskExclusiveSum(const bool* mask_data,
                             int64_t* prefix_sum_data,
                             int64_t n,
                             const phi::Place& place,
                             gpuStream_t stream) {
  // Convert bool mask to int64 for CUB compatibility
  auto mask_int64_alloc = phi::memory_utils::Alloc(place, n * sizeof(int64_t));
  int64_t* mask_int64_data = static_cast<int64_t*>(mask_int64_alloc->ptr());

  int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  BoolToInt64Kernel<<<grid, block, 0, stream>>>(mask_data, mask_int64_data, n);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  phi::Allocator::AllocationPtr allocation;

  // First call to get temp storage size, second call to run the scan
  for (int i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_GPU_SUCCESS(
        cub::DeviceScan::ExclusiveSum(temp_storage,
                                      temp_storage_bytes,
                                      mask_int64_data,
                                      prefix_sum_data,
                                      static_cast<int>(n),
                                      stream));
    if (i == 0 && temp_storage_bytes > 0) {
      allocation = phi::memory_utils::Alloc(place, temp_storage_bytes);
      temp_storage = allocation->ptr();
    }
  }
}

// Asynchronously check that the number of `1` elements present in the mask
// must be <= the number of elements available in `source`.
// This mirrors PyTorch's masked_scatter_size_check kernel: a single-thread
// kernel that avoids any D2H memcpy and stream synchronization.
__global__ void MaskedScatterSizeCheck(const int64_t* mask_exclusive_sum,
                                       const bool* mask,
                                       int64_t srcSize) {
  // Convert exclusive sum to inclusive sum
  const auto totalElements = *mask_exclusive_sum + static_cast<int64_t>(*mask);
  PADDLE_ENFORCE(totalElements <= srcSize,
                 "The number of True elements in mask (%ld) exceeds "
                 "the number of elements in source (%ld).",
                 totalElements,
                 srcSize);
}

template <typename T>
__global__ void MaskedScatterCUDAKernel(const T* x_data,
                                        const bool* mask_data,
                                        const T* value_data,
                                        const int64_t* prefix_sum_data,
                                        const int64_t total,
                                        T* out_data) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  if (mask_data[idx]) {
    out_data[idx] = value_data[prefix_sum_data[idx]];
  } else {
    out_data[idx] = x_data[idx];
  }
}

template <typename T, typename Context>
void MaskedScatterKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& mask,
                         const DenseTensor& value,
                         DenseTensor* out) {
  if (x.numel() == 0 || mask.numel() == 0) {
    dev_ctx.template Alloc<T>(out);
    return;
  }

  auto x_dims = x.dims();
  auto mask_dims = mask.dims();
  auto expanded_size =
      vectorize(funcs::BroadcastTwoDims(x_dims, mask_dims, -1));
  DDim expanded_dims = make_ddim(expanded_size);

  DenseTensor mask_expand;
  DenseTensor x_expand;

  if (mask_dims != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_dims != expanded_dims) {
    ExpandKernel<T, Context>(dev_ctx, x, IntArray(expanded_size), &x_expand);
  } else {
    x_expand = x;
  }

  out->Resize(expanded_dims);
  dev_ctx.template Alloc<T>(out);

  int64_t total = x_expand.numel();
  auto stream = dev_ctx.stream();
  auto* mask_bool_data = mask_expand.data<bool>();

  // Compute exclusive prefix sum of the bool mask -> int64 prefix sum.
  DenseTensor prefix_sum;
  prefix_sum.Resize(mask_expand.dims());
  dev_ctx.template Alloc<int64_t>(&prefix_sum);
  auto* prefix_sum_data = prefix_sum.data<int64_t>();

  MaskExclusiveSum(
      mask_bool_data, prefix_sum_data, total, dev_ctx.GetPlace(), stream);

  // Asynchronously check that the number of `1` elements present in the mask
  // must be <= the number of elements available in `source`.
  MaskedScatterSizeCheck<<<1, 1, 0, stream>>>(
      &prefix_sum_data[total - 1], &mask_bool_data[total - 1], value.numel());

  // Launch masked scatter kernel
  auto config = backends::gpu::GetGpuLaunchConfig1D(dev_ctx, total);
  MaskedScatterCUDAKernel<T>
      <<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
          x_expand.data<T>(),
          mask_bool_data,
          value.data<T>(),
          prefix_sum_data,
          total,
          out->data<T>());
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaskedScatterKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   int16_t,
                   int8_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
