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

#include "paddle/phi/kernels/identity_loss_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/mean_all_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void IdentityLossKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const int reduction,
                        DenseTensor* out) {
  switch (reduction) {
    case 0:
      // sum
      phi::SumRawKernel<T>(
          dev_ctx, x, phi::IntArray({0}), false, true, out->dtype(), out);
      break;
    case 1:
      // mean
      phi::MeanAllKernel<T>(dev_ctx, x, out);
      break;
    case 2:
      // none
      phi::Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);
      break;
    default:
      // error
      PADDLE_THROW(phi::errors::InvalidArgument(
          "reduction should be 0, 1 and 2. But get %d", reduction));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    identity_loss, CPU, ALL_LAYOUT, phi::IdentityLossKernel, float, double) {}
