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
#include "paddle/pten/kernels/hybird/eigen/common.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pten {
namespace eigen {

template <typename DeviceContext, typename T, typename VType>
void fill(const DeviceContext& context, DenseTensor* tensor, VType val) {
  tensor->mutable_data<T>();
  auto t = pten::EigenVector<T>::Flatten(*tensor);
  t.device(*context.eigen_device()) = t.constant(static_cast<T>(val));
}

}  // namespace eigen
}  // namespace pten
