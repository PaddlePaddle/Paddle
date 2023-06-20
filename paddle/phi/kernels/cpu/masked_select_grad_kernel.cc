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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void MaskedSelectGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& mask,
                            const DenseTensor& out_grad,
                            DenseTensor* x_grad) {
  auto input_dim = x.dims();
  auto mask_dim = mask.dims();
  auto out_grad_dims = out_grad.dims();
  auto x_grad_dim = x_grad->dims();

  VLOG(0) << "input_dim:" << input_dim;
  VLOG(0) << "mask_dim:" << mask_dim;
  VLOG(0) << "out_grad_dims:" << out_grad_dims;
  VLOG(0) << "x_grad_dim:" << x_grad_dim;

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

  auto* mask_data = mask.data<bool>();
  auto* input_data = out_grad.data<T>();

  auto* out_data = dev_ctx.template Alloc<T>(x_grad);
  int mask_size = mask.numel();

  int index = 0;
  for (int i = 0; i < mask_size; i++) {
    if (mask_data[i]) {
      out_data[i] = input_data[index];
      index++;
    } else {
      out_data[i] = 0;
    }
  }
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
