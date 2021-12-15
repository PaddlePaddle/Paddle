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

namespace pten {
namespace eigen {

template <typename DevCtx, typename T>
void ElementwiseAdd(const DevCtx& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  out->mutable_data<T>();
  auto eigen_x = pten::EigenVector<T>::Flatten(x);
  auto eigen_y = pten::EigenVector<T>::Flatten(y);
  auto eigen_z = pten::EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();
  eigen_z.device(place) = eigen_x + eigen_y;
}

template <typename DevCtx, typename T>
void ElementwiseSub(const DevCtx& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  auto eigen_x = pten::EigenVector<T>::Flatten(x);
  auto eigen_y = pten::EigenVector<T>::Flatten(y);
  auto eigen_z = pten::EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();
  eigen_z.device(place) = eigen_x - eigen_y;
}

template <typename DevCtx, typename T>
void ElementwiseMul(const DevCtx& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  auto eigen_x = pten::EigenVector<T>::Flatten(x);
  auto eigen_y = pten::EigenVector<T>::Flatten(y);
  auto eigen_z = pten::EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();
  eigen_z.device(place) = eigen_x * eigen_y;
}

}  // namespace eigen
}  // namespace pten
