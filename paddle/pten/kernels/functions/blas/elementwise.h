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

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {
namespace blas {

template <typename DevCtx, typename T>
void ElementwiseAdd(const DevCtx& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
  blas.VADD(x.numel(), x.data<T>(), y.data<T>(), out->mutable_data<T>());
}

template <typename DevCtx, typename T>
void ElementwiseSub(const DevCtx& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  auto blas = paddle::operators::math::GetBlas<DevCtx, T>(dev_ctx);
  blas.VSUB(x.numel(), x.data<T>(), y.data<T>(), out->mutable_data<T>());
}

}  // namespace blas
}  // namespace pten
