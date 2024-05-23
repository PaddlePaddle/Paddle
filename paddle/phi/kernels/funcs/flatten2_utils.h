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
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {
static inline std::vector<int32_t> GetOutputShape(const int axis,
                                                  const phi::DDim &in_dims) {
  if (in_dims.size() == 0) {
    return {1};
  }

  int64_t outer = 1, inner = 1;
  for (int i = 0; i < in_dims.size(); ++i) {
    if (i < axis) {
      if (in_dims[i] == -1 || outer == -1) {
        outer = -1;
      } else {
        outer *= in_dims[i];
      }
    } else {
      if (in_dims[i] == -1 || inner == -1) {
        inner = -1;
      } else {
        inner *= in_dims[i];
      }
    }
  }
  std::vector<int32_t> out_shape(2);
  out_shape[0] = static_cast<int32_t>(outer);
  out_shape[1] = static_cast<int32_t>(inner);
  return out_shape;
}
}  // namespace funcs
}  // namespace phi
