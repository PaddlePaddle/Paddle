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
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  DenseTensor mask_expand;
  if (mask.dims() != x.dims()) {
    auto expanded_size = funcs::MatrixGetBroadcastBatchPortion(
        vectorize(x.dims()), vectorize(mask.dims()));
    ExpandKernel<bool, Context>(
        dev_ctx, mask, IntArray(expanded_size), &mask_expand);
  } else {
    mask_expand = mask;
  }

  auto mask_dim = mask_expand.dims();

  auto input_dim = x.dims();
  auto x_grad_dim = x_grad->dims();

  PADDLE_ENFORCE_EQ(
      input_dim,
      mask_dim,
      phi::errors::InvalidArgument(
          "The dim size of input and mask in OP(masked_selected_grad) "
          "must be equal, but got input dim:(%ld), mask dim: "
          "(%ld). Please check input "
          "value.",
          input_dim,
          mask_dim));

  PADDLE_ENFORCE_EQ(
      input_dim,
      x_grad_dim,
      phi::errors::InvalidArgument(
          "The dim size of input and x_grad in OP(masked_selected_grad) "
          "must be equal, but got input dim:(%ld), x_grad dim: "
          "(%ld). Please check input "
          "value.",
          input_dim,
          x_grad_dim));

  auto* mask_data = mask_expand.data<bool>();
  auto* input_data = out_grad.data<T>();

  auto* out_data = dev_ctx.template Alloc<T>(x_grad);
  int mask_size = mask_expand.numel();

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
      phi::errors::InvalidArgument(
          "The dim size of input and x_grad in OP(masked_selected_grad) "
          "must be equal, but got mask with ones:(%ld), out_grad numel: "
          "(%ld). Please check input "
          "value.",
          index,
          out_grad_numel));
}

}  // namespace phi

PD_REGISTER_KERNEL(masked_select_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaskedSelectGradKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
