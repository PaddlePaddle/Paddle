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

#include "paddle/phi/core/tensor_utils.h"

namespace phi {

static inline int64_t ComputeStartIndex(int64_t start_index, int64_t size) {
  PADDLE_ENFORCE_EQ(
      start_index >= -size && start_index < size,
      true,
      phi::errors::InvalidArgument(
          "The start_index is expected to be in range of [%d, %d), but got %d",
          -size,
          size,
          start_index));
  if (start_index < 0) {
    start_index += size;
  }
  return start_index;
}

}  // namespace phi
