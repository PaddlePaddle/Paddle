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

#include "paddle/phi/common/scalar.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/padding.h"
namespace phi {
template <typename T, typename Context>
void PadKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int>& paddings,
               const Scalar& pad_value,
               bool pad_from_first_axis,
               DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  int rank = x.dims().size();

  // pad the length of paddings to 2*x.ndim
  auto x_dim = x.dims();
  std::vector<int> pad(2 * x_dim.size());
  int paddings_len = paddings.size();
  for (size_t i = 0; i < pad.size(); ++i) {
    int pad_i = static_cast<int>(i) < paddings_len ? paddings[i] : 0;
    pad[i] = pad_i;
  }

  if ((static_cast<int>(paddings_len) == x_dim.size() * 2) &&
      pad_from_first_axis) {
    funcs::PaddingFunctor<Context, T>(
        rank, dev_ctx, pad, pad_value.to<T>(), x, out);
  } else {
    // since PaddingFunctor pad from first axis, if we want to pad from
    // last axis, we need to reverse the paddings
    std::vector<int> pad_reversed(2 * x_dim.size());
    for (int i = 2 * x_dim.size() - 1; i >= 0; --i) {
      int index = 2 * x_dim.size() - 1 - i;
      pad_reversed[i] = (index % 2 == 1) ? pad[index - 1] : pad[index + 1];
    }
    funcs::PaddingFunctor<Context, T>(
        rank, dev_ctx, pad_reversed, pad_value.to<T>(), x, out);
  }
}
}  // namespace phi
