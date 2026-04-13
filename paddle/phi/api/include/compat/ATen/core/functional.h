// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// The file has been adapted from pytorch project
// Licensed under BSD-style license -
// https://github.com/pytorch/pytorch/blob/main/LICENSE

#pragma once

#include <c10/util/ArrayRef.h>
#include <vector>

namespace c10 {

// The passed in function must take T by value (T), or by
// const reference (const T&); taking T by non-const reference
// will result in an error like:
//
//    error: no type named 'type' in 'class std::invoke_result<foobar::__lambda,
//    T>'
//
// No explicit template parameters are required.

// Overload for explicit function and ArrayRef
template <class F, class T>
inline auto fmap(const T& inputs, const F& fn)
    -> std::vector<decltype(fn(*inputs.begin()))> {
  std::vector<decltype(fn(*inputs.begin()))> r;
  r.reserve(inputs.size());
  for (const auto& input : inputs) r.push_back(fn(input));
  return r;
}

// C++ forbids taking an address of a constructor, so here's a workaround...
// Overload for constructor (R) application
template <typename R, typename T>
inline std::vector<R> fmap(const T& inputs) {
  std::vector<R> r;
  r.reserve(inputs.size());
  for (auto& input : inputs) r.push_back(R(input));
  return r;
}

template <typename F, typename T>
inline std::vector<T> filter(at::ArrayRef<T> inputs, const F& fn) {
  std::vector<T> r;
  r.reserve(inputs.size());
  for (auto& input : inputs) {
    if (fn(input)) {
      r.push_back(input);
    }
  }
  return r;
}

template <typename F, typename T>
inline std::vector<T> filter(const std::vector<T>& inputs, const F& fn) {
  return filter<F, T>(static_cast<at::ArrayRef<T>>(inputs), fn);
}

}  // namespace c10
