// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <cstdint>  // for int64_t
#include <numeric>

#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
HOSTDEVICE inline int64_t BinarySearch(const T *x, int64_t num, const T &val) {
  int64_t beg = 0, end = num - 1;
  while (beg <= end) {
    auto mid = ((beg + end) >> 1);
    if (x[mid] == val)
      return mid;
    else if (x[mid] < val)
      beg = mid + 1;
    else
      end = mid - 1;
  }
  return -1;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
