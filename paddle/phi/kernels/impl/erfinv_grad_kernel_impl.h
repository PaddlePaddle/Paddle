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

#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/erfinv_grad_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename Context>
void ErfinvGradKernel(const Context& ctx,
                      const DenseTensor& out,
                      const DenseTensor& out_grad,
                      DenseTensor* x_grad) {
  ctx.template Alloc<phi::ConditionalT<phi::DataType, float, double>>(x_grad);
  auto eigen_out = EigenVector<phi::ConditionalT<phi::DataType, float, double>>::Flatten(out);
  auto eigen_dout = EigenVector<phi::ConditionalT<phi::DataType, float, double>>::Flatten(out_grad);
  auto eigen_dx = EigenVector<phi::ConditionalT<phi::DataType, float, double>>::Flatten(*x_grad);
  auto& place = *ctx.eigen_device();
  constexpr phi::ConditionalT<phi::DataType, float, double> half_sqrt_pi = static_cast<phi::ConditionalT<phi::DataType, float, double>>(1 / M_2_SQRTPI);
  eigen_dx.device(place) = half_sqrt_pi * eigen_dout * eigen_out.square().exp();
}

}  // namespace phi
