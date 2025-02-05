/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/erf_kernel.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename T, typename Context>
void ErfKernel(const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto eigen_out = EigenVector<T>::Flatten(*out);
  auto eigen_in = EigenVector<T>::Flatten(x);
  auto& place = *dev_ctx.eigen_device();
  phi::funcs::EigenErf<std::decay_t<decltype(place)>, T>::Eval(
      place, eigen_out, eigen_in);
}

}  // namespace phi
