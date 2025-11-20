// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

// #The file has been adapted from pytorch project
// #Licensed under  BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/util/Exception.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>

namespace c10 {

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t sum_integers(const C& container) {
  return std::accumulate(
      container.begin(), container.end(), static_cast<int64_t>(0));
}

template <typename Iter,
          std::enable_if_t<std::is_integral_v<
                               typename std::iterator_traits<Iter>::value_type>,
                           int> = 0>
inline int64_t sum_integers(Iter begin, Iter end) {
  return std::accumulate(begin, end, static_cast<int64_t>(0));
}

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  return std::accumulate(container.begin(),
                         container.end(),
                         static_cast<int64_t>(1),
                         std::multiplies<>());
}

template <typename Iter,
          std::enable_if_t<std::is_integral_v<
                               typename std::iterator_traits<Iter>::value_type>,
                           int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_from_dim(const int k, const C& dims) {
  if (k > static_cast<int>(dims.size())) {
    return 1;
  } else {
    auto cbegin = dims.cbegin();
    std::advance(cbegin, k);
    return multiply_integers(cbegin, dims.cend());
  }
}

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_to_dim(const int k, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT((unsigned)k <= dims.size());

  auto cend = dims.cbegin();
  std::advance(cend, k);
  return multiply_integers(dims.cbegin(), cend);
}

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t numelements_between_dim(int k, int l, const C& dims) {
  TORCH_INTERNAL_ASSERT(0 <= k);
  TORCH_INTERNAL_ASSERT(0 <= l);

  if (k > l) {
    std::swap(k, l);
  }

  TORCH_INTERNAL_ASSERT((unsigned)l < dims.size());

  auto cbegin = dims.cbegin();
  auto cend = dims.cbegin();
  std::advance(cbegin, k);
  std::advance(cend, l);
  return multiply_integers(cbegin, cend);
}

}  // namespace c10
