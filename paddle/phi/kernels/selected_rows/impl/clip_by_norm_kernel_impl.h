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

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/core/selected_rows.h"
#include "paddle/phi/kernels/clip_by_norm_kernel.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#include "paddle/phi/kernels/selected_rows/clip_by_norm_kernel.h"

namespace phi {
namespace sr {

template <typename T, typename Context>
void ClipByNormKernel(const Context& dev_ctx,
                      const SelectedRows& x,
                      float max_norm,
                      SelectedRows* out) {
  phi::SelectedRows merged_input;
  phi::funcs::scatter::MergeAdd<Context, T> merge_func;
  merge_func(dev_ctx, x, &merged_input);
  auto input = &(merged_input.value());
  out->set_rows(merged_input.rows());
  out->set_height(merged_input.height());
  auto out_tensor = out->mutable_value();
  out_tensor->Resize(merged_input.value().dims());
  return phi::ClipByNormKernel<T, Context>(
      dev_ctx, *input, max_norm, out_tensor);
}

}  // namespace sr
}  // namespace phi
