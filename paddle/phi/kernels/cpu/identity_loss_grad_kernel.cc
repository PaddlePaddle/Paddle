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

#include "paddle/phi/kernels/identity_loss_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/mean_all_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void IdentityLossGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& out_grad,
                            const int reduction,
                            DenseTensor* x_grad) {
  switch (reduction) {
    case 0:
      // sum
      phi::ReduceSumGradKernel<T>(
          dev_ctx, x, out_grad, std::vector<int64_t>{0}, false, true, x_grad);
      break;
    case 1:
      // mean
      phi::MeanAllGradKernel<T>(dev_ctx, x, out_grad, x_grad);
      break;
    case 2:
      // none
      phi::Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
      break;
    default:
      // error
      PADDLE_THROW(common::errors::InvalidArgument(
          "reduction should be 0, 1 and 2. But get %d", reduction));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(identity_loss_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::IdentityLossGradKernel,
                   float,
                   double) {}
