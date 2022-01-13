/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/kernels/funcs/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pten {

template <typename T, typename Context>
void SignKernel(const Context& dev_ctx,
                const DenseTensor& x,
                DenseTensor* out) {
  out->mutable_data<T>();
  auto eigen_out = pten::EigenVector<T>::Flatten(*out);
  auto eigen_x = pten::EigenVector<T>::Flatten(x);

  auto& dev = *dev_ctx.eigen_device();
  paddle::operators::EigenSign<std::decay_t<decltype(dev)>, T>::Eval(
      dev, eigen_out, eigen_x);
}

}  // namespace pten
