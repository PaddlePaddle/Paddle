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

  if (start < end) {
    PADDLE_ENFORCE_GT(
        step,
        0,
        common::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }

  if (start > end) {
    PADDLE_ENFORCE_LT(step,
                      0,
                      common::errors::InvalidArgument(
                          "The step should be less than 0 while start > end."));
  }

  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

}  // namespace funcs
}  // namespace phi
