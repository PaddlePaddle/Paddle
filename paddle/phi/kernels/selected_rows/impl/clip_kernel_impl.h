// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/kernels/selected_rows/clip_kernel.h"

#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/selected_rows.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void ClipSparseKernel(const Context& dev_ctx,
                      const SelectedRows& x,
                      const Scalar& min,
                      const Scalar& max,
                      SelectedRows* out) {
  auto max_ = max.to<T>();
  auto min_ = min.to<T>();

  PADDLE_ENFORCE_LE(
      min_,
      max_,
      errors::InvalidArgument("max should be greater than or equal to min. "
                              "But received min = %f, max = %f",
                              static_cast<float>(min_),
                              static_cast<float>(max_)));

  PADDLE_ENFORCE_NE(&x,
                    out,
                    errors::InvalidArgument("Inplace clip is not allowed "
                                            "when x is SelectedRows"));
  paddle::operators::math::scatter::MergeAdd<Context, T> merge_func;
  merge_func(dev_ctx, x, out);
  auto* out_tensor = out->mutable_value();
  auto* out_data = out_tensor->data<T>();
  int64_t numel = out_tensor->numel();
  paddle::platform::Transform<Context> trans;
  trans(dev_ctx,
        out_data,
        out_data + numel,
        out_data,
        ClipFunctor<T>(min_, max_));
}
}  // namespace sr
}  // namespace phi
