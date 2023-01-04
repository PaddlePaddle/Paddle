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

#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T>
inline void ShiftAlongDim(T* data,
                          const DDim& input_dim,
                          int64_t dim,
                          int64_t shift) {
  if (dim < 0) {
    dim += input_dim.size();
  }
  if (input_dim[dim] == 0) {
    return;
  }
  shift = shift % input_dim[dim];
  if (shift < 0) {
    shift += input_dim[dim];
  }

  auto outer_loops = 1;
  for (auto i = 0; i < dim; i++) {
    outer_loops *= input_dim[i];
  }
  auto slice_width = 1;
  for (auto i = dim + 1; i < input_dim.size(); i++) {
    slice_width *= input_dim[i];
  }

  VLOG(3) << "shift_along_dim_debug: input_dim: " << input_dim
          << "; dim: " << dim << "; shift: " << shift
          << "; outer_loops: " << outer_loops
          << "; slice_width: " << slice_width;
  if (shift == 0) {
    return;
  }

  std::vector<T> head;
  auto head_size = slice_width * (input_dim[dim] - shift);
  head.resize(head_size);

  for (auto i = 0; i < outer_loops; i++) {
    for (auto j = 0; j < head_size; j++) {
      head[j] = data[i * input_dim[dim] * slice_width + j];
    }
    for (auto j = input_dim[dim] - shift; j < input_dim[dim]; j++) {
      auto dst_pos = j - input_dim[dim] + shift;
      for (auto k = 0; k < slice_width; k++) {
        data[(i * input_dim[dim] + dst_pos) * slice_width + k] =
            data[(i * input_dim[dim] + j) * slice_width + k];
      }
    }
    for (auto j = 0; j < head_size; j++) {
      data[(i * input_dim[dim] + shift) * slice_width + j] = head[j];
    }
  }
}

}  // namespace phi
