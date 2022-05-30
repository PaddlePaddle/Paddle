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

#include "paddle/phi/kernels/label_smooth_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void LabelSmoothGradKernel(const Context& ctx,
                           const DenseTensor& out_grad,
                           float epsilon,
                           DenseTensor* label_grad) {
  ctx.template Alloc<T>(label_grad);
  auto d_out_dim = out_grad.dims()[out_grad.dims().size() - 1];
  if (d_out_dim != 0) {
    auto d_out = EigenVector<T>::Flatten(out_grad);
    auto d_in = EigenVector<T>::Flatten(*label_grad);

    auto& dev = *ctx.eigen_device();
    d_in.device(dev) = static_cast<T>(1 - epsilon) * d_out;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(label_smooth_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::LabelSmoothGradKernel,
                   float,
                   double) {}
