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
#include "paddle/pten/kernels/functions/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pten {
namespace eigen {

template <typename DevCtx, typename T>
void Mean(const DevCtx& dev_ctx, const DenseTensor& x, DenseTensor* out) {
  // TODO(chenweihang): if we design new tensor, we should support
  // the low-level calc functor use new tensor as input,
  // which may be a big project!
  auto eigen_x = pten::EigenVector<T>::Flatten(x);
  auto eigen_out = pten::EigenScalar<T>::From(*out);

  auto& dev = *dev_ctx.eigen_device();
  eigen_out.device(dev) = eigen_x.mean();
}

}  // namespace eigen
}  // namespace pten
