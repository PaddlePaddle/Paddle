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

#include "paddle/phi/kernels/rrelu_grad_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void RReluGradKernel(const Context& dev_ctx,
                     const DenseTensor& mask,
                     const DenseTensor& out_grad,
                     DenseTensor* x_grad) {
  x_grad->mutable_data<T>(dev_ctx.GetPlace());

  auto dX = EigenVector<T>::Flatten(*x_grad);
  auto dY = EigenVector<T>::Flatten(out_grad);
  auto M = EigenVector<T>::Flatten(mask);

  auto& place = *dev_ctx.eigen_device();

  // Can the following be changed to :
  // dX.device(place) = dY * M ;
  // dX.device(place) = dY * M.cast<T>();
  dX.device(place) = dY * M;
}

}  // namespace phi

PD_REGISTER_KERNEL(
    rrelu_grad, 
    CPU, 
    ALL_LAYOUT, 
    phi::RReluGradKernel, 
    float, 
    double,
    phi::dtype::bfloat16) {}
