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

#include "paddle/phi/kernels/select_scatter_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"

namespace phi {

template <typename T>
__global__ void SelectScatterSrcGradGPUKernel(T* src_grad_data,
                                              int index,
                                              int64_t select_index_size,
                                              int64_t outer_dim_size,
                                              int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j;
  i = tid / outer_dim_size;
  j = tid % outer_dim_size;
  int64_t src_offset =
      i * select_index_size * outer_dim_size + index * outer_dim_size + j;
  src_grad_data[src_offset] = 0;
}

template <typename T>
__global__ void SelectScatterValuesGradGPUKernel(T* values_grad_data,
                                                 const T* out_grad_data,
                                                 int index,
                                                 int64_t select_index_size,
                                                 int64_t outer_dim_size,
                                                 int64_t numel) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= numel) return;
  int64_t i, j;
  i = tid / outer_dim_size;
  j = tid % outer_dim_size;
  int64_t src_offset =
      i * select_index_size * outer_dim_size + index * outer_dim_size + j;
  int64_t values_offset = i * outer_dim_size + j;
  values_grad_data[values_offset] = out_grad_data[src_offset];
}

template <typename T, typename Context>
void SelectScatterGradKernel(const Context& dev_ctx,
                             const DenseTensor& src,
                             const DenseTensor& values,
                             const DenseTensor& out_grad,
                             int axis,
                             int index,
                             DenseTensor* src_grad,
                             DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::GPU,
                    true,
                    errors::PreconditionNotMet(
                        "SelectScatterGradOpCUDAKernel only runs on GPU."));

  if (!src_grad && !value_grad) return;
  auto* out_grad_data = out_grad.data<T>();
  auto src_dims = out_grad.dims();
  int64_t select_index_size = src_dims[axis];
  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  for (int i = 0; i < axis; i++) {
    inner_dim_size *= src_dims[i];
  }

  for (int i = axis + 1; i < src_dims.size(); i++) {
    outer_dim_size *= src_dims[i];
  }
  int block = 512;
  int64_t n = inner_dim_size * outer_dim_size;
  int64_t grid = (n + block - 1) / block;
  auto stream = reinterpret_cast<const phi::GPUContext&>(dev_ctx).stream();
  if (src_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, src_grad);
    auto* src_grad_data = src_grad->data<T>();
    SelectScatterSrcGradGPUKernel<T><<<grid, block, 0, stream>>>(
        src_grad_data, index, select_index_size, outer_dim_size, n);
  }
  if (value_grad) {
    value_grad->Resize(values.dims());
    dev_ctx.template Alloc<T>(value_grad);
    auto* value_grad_data = value_grad->data<T>();
    SelectScatterValuesGradGPUKernel<T>
        <<<grid, block, 0, stream>>>(value_grad_data,
                                     out_grad_data,
                                     index,
                                     select_index_size,
                                     outer_dim_size,
                                     n);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(select_scatter_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SelectScatterGradKernel,
                   float,
                   double,
                   int64_t,
                   int,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
