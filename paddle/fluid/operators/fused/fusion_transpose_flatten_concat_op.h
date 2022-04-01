/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <string>
#include <vector>
#include "paddle/phi/core/ddim.h"

namespace paddle {
namespace operators {

inline std::vector<int32_t> GetPermuteShape(const std::vector<int>& axis,
                                            const framework::DDim& in_dims) {
  std::vector<int32_t> out_dims(in_dims.size());
  for (size_t i = 0; i < axis.size(); i++) {
    out_dims[i] = in_dims[axis[i]];
  }
  return out_dims;
}

inline std::vector<int32_t> GetFlattenShape(const int axis,
                                            const std::vector<int>& in_dims) {
  int64_t outer = 1, inner = 1;
  for (int i = 0; i < static_cast<int>(in_dims.size()); ++i) {
    if (i < axis) {
      outer *= in_dims[i];
    } else {
      inner *= in_dims[i];
    }
  }
  std::vector<int32_t> out_shape(2);
  out_shape[0] = outer;
  out_shape[1] = inner;
  return out_shape;
}

}  // namespace operators
}  // namespace paddle
