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

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reverse.h>
#include <thrust/scan.h>

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/kernel_registry.h"
#include "paddle/pten/kernels/masked_select_grad_kernel.h"

namespace pten {

__global__ void SetMaskArrayT(const bool* mask, int32_t* mask_array, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx])
      mask_array[idx] = 1;
    else
      mask_array[idx] = 0;
  }
}

template <typename T>
__global__ void SelectGradWithPrefixMask(const int32_t* mask_prefix_sum,
                                         const bool* mask,
                                         const T* input,
                                         T* out,
                                         int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    if (mask[idx]) {
      int index = mask_prefix_sum[idx];
      out[idx] = input[index];
    } else {
      out[idx] = 0;
    }
  }
}

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& out_grad,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            DenseTensor* x_grad) {
  auto* mask_data = mask.data<bool>();
  auto* input_data = out_grad.data<T>();
  auto* out_data = x_grad->mutable_data<T>(dev_ctx.GetPlace());

  auto input_size = out_grad.numel();
  auto mask_size = mask.numel();
  auto mask_dim = mask.dims();

  auto out_size = mask_size;

  DenseTensor mask_array;
  DenseTensor mask_prefix_sum;
  mask_array.Resize(mask_dim);
  mask_prefix_sum.Resize(mask_dim);

  int32_t* mask_array_data =
      mask_array.mutable_data<int32_t>(dev_ctx.GetPlace());
  int32_t* mask_prefix_sum_data =
      mask_prefix_sum.mutable_data<int32_t>(dev_ctx.GetPlace());
  int threads = 512;
  int grid = (mask_size + threads - 1) / threads;
  auto stream = dev_ctx.stream();
  SetMaskArrayT<<<grid, threads, 0, stream>>>(
      mask_data, mask_array_data, mask_size);

  thrust::device_ptr<int32_t> mask_array_dev_ptr =
      thrust::device_pointer_cast(mask_array_data);
  thrust::device_vector<int32_t> mask_array_vec(mask_array_dev_ptr,
                                                mask_array_dev_ptr + mask_size);
  thrust::exclusive_scan(thrust::device,
                         mask_array_vec.begin(),
                         mask_array_vec.end(),
                         mask_prefix_sum_data);

  SelectGradWithPrefixMask<T><<<grid, threads, 0, stream>>>(
      mask_prefix_sum_data, mask_data, input_data, out_data, mask_size);
}

}  // namespace pten

PT_REGISTER_KERNEL(masked_select_grad,
                   GPU,
                   ALL_LAYOUT,
                   pten::MaskedSelectGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
