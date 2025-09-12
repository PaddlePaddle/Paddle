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

#include "paddle/phi/kernels/hsigmoid_loss_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/hsigmoid_loss_grad.h"

namespace phi {

template <typename T, typename Context>
void HSigmoidLossGradKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& w,
                            const DenseTensor& label,
                            const paddle::optional<DenseTensor>& path,
                            const paddle::optional<DenseTensor>& code,
                            const paddle::optional<DenseTensor>& bias,
                            const DenseTensor& pre_out,
                            const DenseTensor& out_grad,
                            int num_classes,
                            bool is_sparse,
                            DenseTensor* x_grad,
                            DenseTensor* w_grad,
                            DenseTensor* bias_grad) {
  HSigmoidLossGradKernelImpl<T>(dev_ctx,
                                x,
                                w,
                                label,
                                path,
                                code,
                                bias,
                                pre_out,
                                out_grad,
                                num_classes,
                                is_sparse,
                                x_grad,
                                w_grad,
                                bias_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(hsigmoid_loss_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::HSigmoidLossGradKernel,
                   float,
                   double) {}
