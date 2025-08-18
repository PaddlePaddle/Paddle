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
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace funcs {

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  PADDLE_ENFORCE_NE(
      step,
      0,
      common::errors::InvalidArgument("The step of range op should not be 0."));
  if constexpr (std::is_same_v<T, phi::dtype::bfloat16> ||
                std::is_same_v<T, phi::dtype::float16>) {
    PADDLE_ENFORCE_EQ(phi::dtype::isfinite(start) && phi::dtype::isfinite(end),
                      true,
                      common::errors::InvalidArgument(
                          "The start and end of range op should be finite "
                          "numbers, but received  %f -> %f.",
                          static_cast<double>(start),
                          static_cast<double>(end)));
  } else if constexpr (std::is_floating_point_v<T>) {
    PADDLE_ENFORCE_EQ(std::isfinite(start) && std::isfinite(end),
                      true,
                      common::errors::InvalidArgument(
                          "The start and end of range op should be finite "
                          "numbers, but received  %f -> %f.",
                          static_cast<double>(start),
                          static_cast<double>(end)));
  }
  if (start < end) {
    if (step < 0) {
      *size = 0;
      return;
    }
  }

  if (start > end) {
    if (step > 0) {
      *size = 0;
      return;
    }
  }

  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

}  // namespace funcs
}  // namespace phi
