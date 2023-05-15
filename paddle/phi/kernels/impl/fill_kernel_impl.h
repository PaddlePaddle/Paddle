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

#include "paddle/phi/kernels/fill_kernel.h"

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void FillKernel(const Context& dev_ctx,
                const DenseTensor& x UNUSED,
                const Scalar& value,
                DenseTensor* out) {
  T fill_var = value.to<T>();

  PADDLE_ENFORCE_EQ(std::isnan(static_cast<double>(fill_var)),
                    false,
                    phi::errors::InvalidArgument("fill value should not be NaN,"
                                                 " but received NaN"));

  dev_ctx.template Alloc<T>(out);

  phi::funcs::SetConstant<Context, T> functor;
  functor(dev_ctx, out, fill_var);
}

}  // namespace phi
