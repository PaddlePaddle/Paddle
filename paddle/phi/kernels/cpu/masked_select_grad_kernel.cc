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

#include "paddle/phi/kernels/masked_select_grad_kernel.h"

#include "paddle/phi/kernels/expand_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/empty_kernel.h"
#include "paddle/phi/kernels/expand_grad_kernel.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  // x_grad.size() == x.size()
  // x.size() == mask.size(), no broadcast, expand_mask = false, expand_x =
  // false x.size() < mask.size(), x broadcast to mask, expand_mask = false,
  // expand_x = true x.size() > mask.size(), mask broadcast to x, expand_mask =
  // true, expand_x = false
  DenseTensor mask_expand;
  DenseTensor x_grad_expand;
  bool expand_x = false;

  auto expanded_size = funcs::MatrixGetBroadcastBatchPortion(
      common::vectorize(x_grad->dims()), common::vectorize(mask.dims()));
  auto expanded_dims = common::make_ddim(expanded_size);

  if (mask.dims() != expanded_dims) {
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  if (x_grad->dims() != expanded_dims) {
    x_grad_expand = Empty<T, Context>(dev_ctx, IntArray(expanded_size));
    expand_x = true;
  } else {
    expand_x = false;
  }

  auto* out_data = dev_ctx.template Alloc<T>(x_grad);
  if (expand_x) {
    out_data = x_grad_expand.data<T>();
  }

  auto* mask_data = mask_expand.data<bool>();
  auto* input_data = out_grad.data<T>();
  int mask_size = static_cast<int>(mask_expand.numel());

  int index = 0;
  for (int i = 0; i < mask_size; i++) {
    if (mask_data[i]) {
      out_data[i] = input_data[index];
      index++;
    } else {
      out_data[i] = 0;
    }
  }

  auto out_grad_numel = out_grad.numel();
  PADDLE_ENFORCE_EQ(
      index,
      out_grad_numel,
      common::errors::InvalidArgument(
          "The dim size of input and x_grad in OP(masked_selected_grad) "
          "must be equal, but got mask with ones:(%ld), out_grad numel: "
          "(%ld). Please check input "
          "value.",
          index,
          out_grad_numel));

  if (expand_x) {
    ExpandGradKernel<T, Context>(
        dev_ctx, x, x_grad_expand, IntArray(expanded_size), x_grad);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
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
                   phi::dtype::complex<double>) {}
