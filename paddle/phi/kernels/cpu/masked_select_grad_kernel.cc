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
  auto* mask_data = mask.data<bool>();
  auto* input_data = out_grad.data<T>();
  auto* out_data = x_grad->mutable_data<T>(dev_ctx.GetPlace());
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
