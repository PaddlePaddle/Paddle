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

#include "paddle/phi/kernels/masked_fill_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/common_infer_shape_functions.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void MaskedFillKernel(const Context& dev_ctx,
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
      common::vectorize(phi::funcs::BroadcastTwoDims(x_dims, mask_dims, -1));

  DenseTensor mask_expand;
  DenseTensor x_expand;
  DenseTensor value_expand;

  DDim expand_dims = common::make_ddim(expanded_size);
  if (mask.dims() != expand_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x.dims() != expand_dims) {
    ExpandKernel<T, Context>(dev_ctx, x, IntArray(expanded_size), &x_expand);
  } else {
    x_expand = x;
  }

  if (value.dims() != expand_dims && value.numel() != 1) {
    ExpandKernel<T, Context>(
        dev_ctx, value, IntArray(expanded_size), &value_expand);
  } else {
    value_expand = value;
  }

  auto input_data = x_expand.data<T>();
  auto mask_data = mask_expand.data<bool>();
  auto value_data = value_expand.data<T>();

  auto x_size = x_expand.numel();

  out->Resize(expand_dims);

  auto out_data = dev_ctx.template HostAlloc<T>(out);
  if (value.numel() == 1) {
#pragma unroll
    for (int i = 0; i < x_size; i++) {
      if (mask_data[i]) {
        out_data[i] = value_data[0];
      } else {
        out_data[i] = input_data[i];
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < x_size; i++) {
      if (mask_data[i]) {
        out_data[i] = value_data[i];
      } else {
        out_data[i] = input_data[i];
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_fill,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedFillKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {
  kernel->InputAt(1).SetDataType(phi::DataType::BOOL);
}
