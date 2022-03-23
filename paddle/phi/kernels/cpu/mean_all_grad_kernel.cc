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

#include "paddle/phi/kernels/mean_all_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& out_grad,
                       DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(out_grad.numel(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Mean Gradient should be scalar. But received "
                        "Out@Grad's elements num is %d.",
                        out_grad.numel()));
  dev_ctx.template Alloc<T>(x_grad);

  T ig_size = static_cast<T>(x_grad->numel());
  Eigen::DSizes<int, 1> bcast(static_cast<int>(ig_size));
  EigenVector<T>::Flatten(*x_grad).device(*dev_ctx.eigen_device()) =
      (EigenVector<T>::From(out_grad) / ig_size).broadcast(bcast);
}

}  // namespace phi

PD_REGISTER_KERNEL(mean_all_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::MeanAllGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
