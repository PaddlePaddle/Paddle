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
void FillAnyKernel(const DeviceContext& dev_ctx,
                   float value_float,
                   int value_int,
                   DenseTensor* out) {
  auto isfloat = ((typeid(float) == typeid(T)) ||
                  (typeid(double) == typeid(T) ||
                   typeid(paddle::platform::float16) == typeid(T)));

  T fill_var = static_cast<T>(value_float);
  if (!isfloat) {
    fill_var = static_cast<T>(value_int);
  }

  PADDLE_ENFORCE_EQ(std::isnan(static_cast<double>(fill_var)),
                    false,
                    paddle::platform::errors::InvalidArgument(
                        "fill value should not be NaN, but received NaN"));

  dev_ctx.template Alloc<T>(out);

  pten::funcs::SetConstant<DeviceContext, T> functor;
  functor(dev_ctx, out, static_cast<T>(fill_var));
}

}  // namespace pten
