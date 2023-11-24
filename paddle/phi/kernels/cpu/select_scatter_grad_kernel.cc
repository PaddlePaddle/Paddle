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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void SelectScatterGradKernel(const Context& dev_ctx,
                             const DenseTensor& src,
                             const DenseTensor& values,
                             const DenseTensor& out_grad,
                             int axis,
                             int index,
                             DenseTensor* src_grad,
                             DenseTensor* value_grad) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
                    true,
                    errors::PreconditionNotMet(
                        "SelectScatterGradOpKernel only runs on CPU."));
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
  if (src_grad) {
    phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, src_grad);
    auto* src_grad_data = src_grad->data<T>();
    int64_t src_offset = 0;
    for (int i = 0; i < inner_dim_size; i++) {
      src_offset =
          i * select_index_size * outer_dim_size + index * outer_dim_size;
      memset(src_grad_data + src_offset, 0, sizeof(T) * outer_dim_size);
    }
  }
  if (value_grad) {
    value_grad->Resize(values.dims());
    dev_ctx.template Alloc<T>(value_grad);
    auto* values_grad_data = value_grad->data<T>();
    int64_t values_offset = 0, src_offset = 0;
    for (int i = 0; i < inner_dim_size; i++) {
      src_offset =
          i * select_index_size * outer_dim_size + index * outer_dim_size;
      values_offset = i * outer_dim_size;
      memcpy(values_grad_data + values_offset,
             out_grad_data + src_offset,
             sizeof(T) * outer_dim_size);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(select_scatter_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::SelectScatterGradKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t) {}
