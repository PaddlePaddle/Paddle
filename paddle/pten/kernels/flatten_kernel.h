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
#include "paddle/pten/infermeta/unary.h"
#include "paddle/pten/kernels/empty_kernel.h"

namespace pten {

template <typename T, typename Context>
void FlattenKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   int start_axis,
                   int stop_axis,
                   DenseTensor* out);

template <typename T, typename Context>
void FlattenWithXShape(const Context& dev_ctx,
                       const DenseTensor& x,
                       int start_axis,
                       int stop_axis,
                       DenseTensor* out,
                       DenseTensor* xshape);

template <typename T, typename Context>
DenseTensor Flatten(const Context& dev_ctx,
                    const DenseTensor& x,
                    int start_axis,
                    int stop_axis) {
  auto dense_out = Empty<T, Context>(dev_ctx);
  MetaTensor meta_out(&dense_out);
  FlattenInferMeta(x, start_axis, stop_axis, &meta_out);
  FlattenKernel<T, Context>(dev_ctx, x, start_axis, stop_axis, &dense_out);
  return dense_out;
}

}  // namespace pten
