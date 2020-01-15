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
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace detail {
/**
 * Get Reference From Pointer with check. The error message is printf format,
 * and passed by `args`
 */
template <typename T, typename... ARGS>
inline T& Ref(T* ptr, ARGS&&... args) {
  PADDLE_ENFORCE_NOT_NULL(ptr, ::paddle::string::Sprintf(args...));
  return *ptr;
}

template <typename T, typename... ARGS>
inline std::vector<std::reference_wrapper<T>> VectorRef(
    const std::vector<T*>& vec, ARGS&&... args) {
  std::vector<std::reference_wrapper<T>> result;
  result.reserve(vec.size());
  for (auto* ptr : vec) {
    result.emplace_back(Ref(ptr, args...));
  }
  return result;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
