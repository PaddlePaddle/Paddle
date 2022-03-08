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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void ShapeKernel(const Context& ctx,
                 const DenseTensor& input,
                 DenseTensor* out) {
  auto in_var = &input;
  phi::DDim in_dims;
  in_dims = in_var->dims();
  auto out_t = out;
  out_t->Resize({in_dims.size()});
  auto out_data = ctx.template HostAlloc<int32_t>(out_t);
  for (int i = 0; i < in_dims.size(); ++i) {
    out_data[i] = in_dims[i];
  }
}

}  // namespace phi
