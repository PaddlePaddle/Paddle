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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

template <typename T>
std::vector<T> ComputeDimStride(const std::vector<T> dim) {
  size_t dim_size = dim.size();
  std::vector<T> dim_strides;
  dim_strides.resize(dim_size);
  for (size_t i = 0; i < dim_size - 1; i++) {
    size_t temp_stride = 1;
    for (size_t j = i + 1; j < dim_size; j++) {
      temp_stride = temp_stride * dim[j];
    }
    dim_strides[i] = temp_stride;
  }
  dim_strides[dim_size - 1] = 1;
  return dim_strides;
}

}  // namespace pten
