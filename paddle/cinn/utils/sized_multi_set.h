// Copyright (c) 2022 CINN Authors. All Rights Reserved.
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

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <set>
#include "paddle/common/enforce.h"

namespace cinn {
namespace utils {

/**
 * A data structure stores limited size ordered duplicatable elements.
 *
 * The default implementation would pop maximal element when size reaches
 * capacity. Users could change pop_max_when_full parameter of constructor
 * to false to pop minimal element.
 *
 * The underneath implementation uses std::multiset
 */
template <class T,
          class Compare = std::less<T>,
          class Alloc = std::allocator<T>>
class SizedMultiSet {
 public:
  explicit SizedMultiSet(size_t capacity, bool pop_max_when_full = true)
      : capacity_(capacity), pop_max_when_full_(pop_max_when_full) {}

  void Push(const T& data) {
    multi_set_.insert(data);
    if (multi_set_.size() > capacity_) {
      Pop();
    }
  }

  void Push(T&& data) {
    multi_set_.insert(data);
    if (multi_set_.size() > capacity_) {
      Pop();
    }
  }

  void Pop() {
    PADDLE_ENFORCE_GE(multi_set_.size(),
                      1UL,
                      ::common::errors::PreconditionNotMet(
                          "Call Pop on empty SizedMultiSet."));
    if (pop_max_when_full_) {
      multi_set_.erase(--multi_set_.end());
    } else {
      multi_set_.erase(multi_set_.begin());
    }
  }

  T MaxValue() const { return *(multi_set_.rbegin()); }

  T MinValue() const { return *(multi_set_.begin()); }

  size_t Size() const { return multi_set_.size(); }

  template <class ContainerType>
  ContainerType ReturnAsContainer() const {
    return ContainerType(multi_set_.begin(), multi_set_.end());
  }

 private:
  size_t capacity_;
  bool pop_max_when_full_;
  std::multiset<T, Compare, Alloc> multi_set_;
};

}  // namespace utils
}  // namespace cinn
