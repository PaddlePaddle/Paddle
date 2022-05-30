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

#include <type_traits>
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace memory {

class Buffer {
 public:
  explicit Buffer(const platform::Place &place) : place_(place) {}

  template <typename T>
  T *Alloc(size_t size) {
    using AllocT = typename std::conditional<std::is_same<T, void>::value,
                                             uint8_t, T>::type;
    if (UNLIKELY(size == 0)) return nullptr;
    size *= sizeof(AllocT);
    if (allocation_ == nullptr || allocation_->size() < size) {
      allocation_ = memory::Alloc(place_, size);
    }
    return reinterpret_cast<T *>(allocation_->ptr());
  }

  template <typename T>
  const T *Get() const {
    return reinterpret_cast<const T *>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  template <typename T>
  T *GetMutable() {
    return reinterpret_cast<T *>(
        allocation_ && allocation_->size() > 0 ? allocation_->ptr() : nullptr);
  }

  size_t Size() const { return allocation_ ? 0 : allocation_->size(); }

  platform::Place GetPlace() const { return place_; }

 private:
  AllocationPtr allocation_;
  platform::Place place_;
};

}  // namespace memory
}  // namespace paddle
