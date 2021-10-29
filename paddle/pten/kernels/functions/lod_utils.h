// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/platform/errors.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

static inline int ComputeAxis(int axis, int rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      paddle::platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

static inline pten::DDim ComputeAndCheckShape(
    const std::vector<pten::DDim>& inputs_dims, const size_t axis) {
  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    PADDLE_ENFORCE_EQ(inputs_dims[i].size(),
                      out_dims.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The shape of input[0] and input[%d] "
                          "is expected to be equal."
                          "But received input[0]'s shape = "
                          "[%s], input[%d]'s shape = [%s].",
                          i,
                          inputs_dims[0],
                          i,
                          inputs_dims[i]));
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        out_dims[axis] += inputs_dims[i][j];
      } else {
        bool check_shape = inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0;
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j],
                            inputs_dims[i][j],
                            paddle::platform::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j,
                                i,
                                inputs_dims[0],
                                i,
                                inputs_dims[i]));
        }
      }
    }
  }
  return out_dims;
}

LoD ConvertToLengthBasedLoD(const LoD& offset_lod);
void AppendLoD(LoD* lod, const LoD& lod_length);

}  // namespace pten
