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

#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/nanmedian_grad_kernel.h"

namespace phi {

template <typename T, typename Context>
void PostprocessMedianGradKernel(const Context& dev_ctx,
                                 DenseTensor* input,
                                 const IntArray& raw_axes,
                                 DenseTensor* x) {
  auto input_dim = input->dims();
  auto rank = input_dim.size();

  std::vector<int64_t> axes = raw_axes.GetData();
  int64_t axes_size = static_cast<int>(axes.size());
  for (int64_t i = 0; i < axes_size; i++) {
    if (axes[i] < 0) {
      axes[i] += rank;
    }
  }

  std::vector<int> trans_back;
  std::vector<int> reshape_back;
  trans_back.reserve(rank);
  trans_back.resize(rank);

  int offset = 0;
  for (int64_t i = 0; i < rank; i++) {
    if (std::find(axes.begin(), axes.end(), i) == axes.end()) {
      reshape_back.push_back(input_dim[i]);
      trans_back[i] = offset;
      offset += 1;
    }
  }

  for (int64_t i = 0; i < rank; i++) {
    if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
      trans_back[i] = offset;
      reshape_back.push_back(input_dim[i]);
      offset += 1;
    }
  }

  input->Resize(make_ddim(reshape_back));
  funcs::TransCompute<Context, T>(
      static_cast<int>(trans_back.size()), dev_ctx, *input, x, trans_back);
}

}  // namespace phi
