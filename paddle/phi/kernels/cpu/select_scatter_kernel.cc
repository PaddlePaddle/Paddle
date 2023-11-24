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

#include "paddle/phi/kernels/select_scatter_kernel.h"
#include "glog/logging.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"

namespace phi {

template <typename T, typename Context>
void SelectScatterKernel(const Context& dev_ctx,
                         const DenseTensor& src,
                         const DenseTensor& values,
                         int axis,
                         int index,
                         DenseTensor* out) {
  PADDLE_ENFORCE_EQ(
      dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
      true,
      errors::PreconditionNotMet("SelectScatterKernel only runs on CPU."));
  phi::Copy(dev_ctx, src, dev_ctx.GetPlace(), false, out);
  auto* values_data = values.data<T>();
  auto* out_data = out->data<T>();
  int64_t src_size = src.numel();
  int64_t values_size = values.numel();
  auto src_dims = src.dims();
  if (src_size == 0 || values_size == 0) {
    VLOG(3) << "zero size input found";
    phi::errors::InvalidArgument("src_size, values_size cannot be 0");
    return;
  }
  int64_t select_index_size = src_dims[axis];
  int64_t inner_dim_size = 1;
  int64_t outer_dim_size = 1;
  for (int i = 0; i < axis; i++) {
    inner_dim_size *= src_dims[i];
  }

  for (int i = axis + 1; i < src_dims.size(); i++) {
    outer_dim_size *= src_dims[i];
  }
  int64_t src_offset = 0, values_offset = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    src_offset =
        i * select_index_size * outer_dim_size + index * outer_dim_size;
    values_offset = i * outer_dim_size;
    memcpy(out_data + src_offset,
           values_data + values_offset,
           sizeof(T) * outer_dim_size);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(select_scatter,
                   CPU,
                   ALL_LAYOUT,
                   phi::SelectScatterKernel,
                   float,
                   double,
                   int,
                   uint8_t,
                   int64_t) {}
