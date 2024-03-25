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

#include <algorithm>
#include <cstdint>  // for int64_t
#include <numeric>

#include "paddle/common/hostdevice.h"

namespace phi {
namespace funcs {

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

template <typename T1, typename T2>
HOSTDEVICE inline size_t LowerBound(const T1 *x, size_t num, const T2 &val) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)  // @{ Group LowerBound
  // The following code is from
  // https://en.cppreference.com/w/cpp/algorithm/lower_bound
  auto *first = x;
  int64_t count = static_cast<int64_t>(num);
  while (count > 0) {
    int64_t step = (count >> 1);
    auto *it = first + step;
    if (*it < val) {
      first = ++it;
      count -= (step + 1);
    } else {
      count = step;
    }
  }
  return static_cast<size_t>(first - x);
#else
  return static_cast<size_t>(std::lower_bound(x, x + num, val) - x);
#endif  // @} End Group LowerBound
}

template <typename T1, typename T2>
HOSTDEVICE inline size_t UpperBound(const T1 *x, size_t num, const T2 &val) {
#if defined(__CUDA_ARCH__) || defined(__HIPCC__)  // @{ Group UpperBound
  // The following code is from
  // https://en.cppreference.com/w/cpp/algorithm/upper_bound
  auto *first = x;
  int64_t count = static_cast<int64_t>(num);
  while (count > 0) {
    auto step = (count >> 1);
    auto *it = first + step;
    if (val < *it) {
      count = step;
    } else {
      first = ++it;
      count -= (step + 1);
    }
  }
  return static_cast<size_t>(first - x);
#else
  return static_cast<size_t>(std::upper_bound(x, x + num, val) - x);
#endif  // @} End Group UpperBound
}

}  // namespace funcs
}  // namespace phi
