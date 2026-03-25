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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"

namespace phi {

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
  auto* out_data = dev_ctx.template Alloc<T>(out);
  auto* x_data = x_expand.data<T>();
  auto* mask_data = mask_expand.data<bool>();
  auto* value_data = value.data<T>();
  int64_t total = x_expand.numel();
  int64_t value_numel = value.numel();

  int64_t count = 0;
  for (int64_t i = 0; i < total; i++) {
    if (mask_data[i]) {
      PADDLE_ENFORCE_LT(
          count,
          value_numel,
          common::errors::InvalidArgument(
              "Number of True values in mask (%d) exceeds the number of "
              "elements in value (%d).",
              count + 1,
              value_numel));
      out_data[i] = value_data[count];
      count++;
    } else {
      out_data[i] = x_data[i];
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_scatter,
                   CPU,
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
