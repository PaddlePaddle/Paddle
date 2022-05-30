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
#include <utility>
#include <vector>
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/padding.h"
namespace phi {
template <typename T, typename Context>
void PadKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int>& paddings,
               float pad_value,
               DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  int rank = x.dims().size();
  funcs::PaddingFunctor<Context, T>(
      rank, dev_ctx, paddings, static_cast<T>(pad_value), x, out);
}
}  // namespace phi
