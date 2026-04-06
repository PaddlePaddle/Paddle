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

#include <memory>
#include <type_traits>

#include "c10/util/Exception.h"

namespace c10 {

// Compatibility wrapper around a raw pointer so that existing code
// written to deal with a shared_ptr can keep working.
template <typename T>
class SingletonTypePtr {
 public:
  SingletonTypePtr(T* p) : repr_(p) {}  // NOLINT(runtime/explicit)

  // We need this to satisfy Pybind11, but it shouldn't be hit.
  explicit SingletonTypePtr(std::shared_ptr<T> /*unused*/) {
    TORCH_CHECK(false);
  }

  using element_type = typename std::shared_ptr<T>::element_type;

  template <typename U = T,
            std::enable_if_t<!std::is_same_v<std::remove_const_t<U>, void>,
                             bool> = true>
  T& operator*() const {
    return *repr_;
  }

  T* get() const { return repr_; }

  T* operator->() const { return repr_; }

  operator bool() const { return repr_ != nullptr; }

 private:
  T* repr_{nullptr};
};

template <typename T, typename U>
bool operator==(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return static_cast<const void*>(lhs.get()) ==
         static_cast<const void*>(rhs.get());
}

template <typename T, typename U>
bool operator!=(SingletonTypePtr<T> lhs, SingletonTypePtr<U> rhs) {
  return !(lhs == rhs);
}

}  // namespace c10
