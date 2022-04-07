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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/strided_memcpy.h"
#include "paddle/fluid/operators/utils.h"

#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/funcs/concat_funcs.h"

namespace paddle {
namespace operators {

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank, true,
      platform::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d", -rank,
          rank, axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

}  // namespace operators
}  // namespace paddle
