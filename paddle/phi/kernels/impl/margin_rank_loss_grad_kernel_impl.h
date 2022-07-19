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

#include "paddle/phi/kernels/margin_rank_loss_grad_kernel.h"

#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
//#include "paddle/fluid/framework/eigen.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T, typename Context>
void MarginRankLossGradKernel(const Context& dev_ctx,
                              const DenseTensor& label,
                              const DenseTensor& activated,
                              const DenseTensor& out_grad,
                              float margin,
                              DenseTensor* left_grad,
                              DenseTensor* right_grad) {
  auto d_out = phi::EigenVector<T>::Flatten(out_grad);
  auto act = phi::EigenVector<T>::Flatten(activated);
  auto label_t = phi::EigenVector<T>::Flatten(label);

  auto& dev = *dev_ctx.eigen_device();
  // compute left_grad
  if (left_grad) {
    dev_ctx.template Alloc<T>(left_grad);
    auto d_x1 = phi::EigenVector<T>::Flatten(*left_grad);
    d_x1.device(dev) = -d_out * act * label_t;
  }
  // compute right_grad
  if (right_grad) {
    dev_ctx.template Alloc<T>(right_grad);
    auto d_x2 = phi::EigenVector<T>::Flatten(*right_grad);
    d_x2.device(dev) = d_out * act * label_t;
  }
}
}  // namespace phi
