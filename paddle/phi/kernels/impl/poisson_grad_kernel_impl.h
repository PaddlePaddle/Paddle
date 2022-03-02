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

#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/poisson_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void PoissonGradKernel(const Context& ctx, DenseTensor* x_grad) {
  ctx.template Alloc<T>(x_grad);
  phi::funcs::SetConstant<Context, T> functor;
  functor(ctx, x_grad, static_cast<T>(0));
}

}  // namespace phi
