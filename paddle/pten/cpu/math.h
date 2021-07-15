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

#include "paddle/pten/core/base_tensor.h"
#include "paddle/pten/module/sign.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/platform/device_context.h"

namespace pt {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenScalar = paddle::framework::EigenScalar<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = paddle::framework::EigenVector<T, MajorType, IndexType>;

using CPUDeviceContext = paddle::platform::CPUDeviceContext;

template <typename T>
void Sign(const CPUDeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  module::Sign<CPUDeviceContext, T>(dev_ctx, x, out);
}

template <typename T>
void Mean(const CPUDeviceContext& dev_ctx,
          const BaseTensor& x,
          BaseTensor* out) {
  out->mutable_data<T>();
  auto x_data = EigenVector<T>::Flatten(x);
  auto y_data = EigenScalar<T>::From(*out);
  auto& place = *dev_ctx.eigen_device();
  y_data.device(place) = x_data.mean();
}

}  // namespace pt
