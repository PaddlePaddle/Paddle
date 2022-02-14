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

#include "paddle/pten/kernels/funcs/math_function.h"

namespace pten {

template <typename T, typename DeviceContext>
void FillAnyGradKernel(const DeviceContext& dev_ctx, DenseTensor* x_grad) {
  if (x_grad) {
    dev_ctx.template Alloc<T>(x_grad);
    pten::funcs::SetConstant<DeviceContext, T> functor;
    functor(dev_ctx, x_grad, T(0));
  }
}

}  // namespace pten
