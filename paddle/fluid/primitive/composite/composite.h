// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>
#include <vector>

#include "paddle/fluid/primitive/primitive/primitive.h"
#include "paddle/fluid/primitive/type/lazy_tensor.h"
#include "paddle/fluid/primitive/utils/utils.h"

namespace paddle {
namespace primitive {
namespace details {

template <typename T>
Tensor mean_decomp(const Tensor& x, const IntArray& axis, bool keepdim) {
  std::cout << "******** mean decomp begin ********" << std::endl;
  std::vector<int64_t> x_dim = phi::vectorize<int64_t>(x.dims());
  int64_t axis_size = axis.size();
  int64_t x_dim_size = x_dim.size();
  auto axis_ = std::vector<int64_t>();
  if (axis_size == 0) {
    for (int64_t i = 0; i < x_dim_size; i++) {
      axis_.push_back(i);
    }
  } else {
    axis_ = axis.GetData();
    for (int64_t i = 0; i < axis_size; i++) {
      if (axis[i] < 0) {
        axis_[i] = axis[i] + x_dim_size;
      }
    }
  }
  std::cout << "******** mean decomp 1 ********" << std::endl;

  int64_t value = 1;
  for (size_t i = 0; i < axis_.size(); i++) {
    value *= x_dim[axis_[i]];
  }
  auto sum_x = sum<T>(x, IntArray(axis_), x.dtype(), keepdim);
  auto res = divide<T>(
      sum_x, full<T>(phi::vectorize(sum_x.dims()), value, sum_x.dtype()));
  std::cout << "******** mean decomp end ******** value " << value << std::endl;

  return res;
}

}  // namespace details

}  // namespace primitive
}  // namespace paddle
