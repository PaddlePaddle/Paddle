// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void ShuffleChannelOpKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            int group,
                            DenseTensor* out) {
  const auto& input_dims = x.dims();
  auto num = input_dims[0];
  auto channel = input_dims[1];
  auto height = input_dims[2];
  auto weight = input_dims[3];

  auto feature_map_size = channel * height * weight;
  auto sp_sz = height * weight;
  int group_row = group;
  int group_column = channel / group_row;

  const T* input_data = x.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(out);
  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < group_row; ++i) {
      for (int j = 0; j < group_column; ++j) {
        const T* p_i =
            input_data + n * feature_map_size + (i * group_column + j) * sp_sz;
        T* p_o =
            output_data + n * feature_map_size + (j * group_row + i) * sp_sz;
        memcpy(p_o, p_i, sizeof(int) * sp_sz);
      }
    }
  }
}

template <typename T, typename Context>
void ShuffleChannelGradOpKernel(const Context& dev_ctx,
                                const DenseTensor& out_grad,
                                int group,
                                DenseTensor* x_grad) {
  const auto& input_dims = x_grad->dims();
  auto num = input_dims[0];
  auto channel = input_dims[1];
  auto height = input_dims[2];
  auto weight = input_dims[3];
  auto feature_map_size = channel * height * weight;
  auto sp_sz = height * weight;

  int group_row = group;
  int group_column = channel / group_row;

  T* input_grad_data = dev_ctx.template Alloc<T>(x_grad);
  const T* output_grad_data = out_grad.data<T>();
  for (int n = 0; n < num; ++n) {
    for (int i = 0; i < group_row; ++i) {
      for (int j = 0; j < group_column; ++j) {
        const T* p_i = output_grad_data + n * feature_map_size +
                       (i * group_column + j) * sp_sz;
        T* p_o = input_grad_data + n * feature_map_size +
                 (j * group_row + i) * sp_sz;
        memcpy(p_o, p_i, sizeof(int) * sp_sz);
      }
    }
  }
}
}  // namespace phi
