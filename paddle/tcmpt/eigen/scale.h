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

#include "paddle/tcmpt/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

namespace pt {
namespace module {

template <typename DevCtx, typename T>
void Scale(const DevCtx& dev_ctx,
           const DenseTensor& x,
           float scale,
           float bias,
           bool bias_after_scale,
           DenseTensor* out) {
  // calc
  out->mutable_data<T>();
  auto eigen_out = paddle::framework::EigenVector<T>::Flatten(*out);
  auto eigen_x = paddle::framework::EigenVector<T>::Flatten(x);
  auto& dev = *dev_ctx.eigen_device();
  // TODO(chenweihang): now the eigen function here need the dtype of scale,
  // eigen_x, bias should be same, so here need cast for two scalar arg,
  // maybe we declare that the type of scale and bias is T?
  paddle::operators::EigenScale<std::decay_t<decltype(dev)>, T>::Eval(
      dev,
      eigen_out,
      eigen_x,
      static_cast<T>(scale),
      static_cast<T>(bias),
      bias_after_scale);
}

}  // namespace module
}  // namespace pt
