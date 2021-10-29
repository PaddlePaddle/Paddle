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

LoD ConvertToLengthBasedLoD(const LoD& offset_lod);
void AppendLoD(LoD* lod, const LoD& lod_length);

}  // namespace pten
