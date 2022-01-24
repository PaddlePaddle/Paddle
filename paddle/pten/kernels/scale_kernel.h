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

#include "paddle/pten/common/scalar.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/empty_kernel.h"
namespace pten {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const Scalar& scale,
                 float bias,
                 bool bias_after_scale,
                 DenseTensor* out);

template <typename T, typename Context>
DenseTensor Scale(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& scale,
                  float bias,
                  bool bias_after_scale) {
  auto out_meta = UnchangedInferMeta(x.meta());
  auto dense_out = pten::Empty(dev_ctx, std::move(out_meta));
  ScaleKernel<T, Context>(
      dev_ctx, x, scale, bias, bias_after_scale, &dense_out);
  return dense_out;
}

}  // namespace pten
