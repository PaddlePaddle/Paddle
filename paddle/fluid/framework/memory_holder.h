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

#include <cstdint>
#include <cstring>
#include <memory>
#include <typeindex>
#include <vector>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace internal {

class MemoryHolder {
 private:
  struct InternalHolder {
    InternalHolder(size_t size, const platform::Place &place)
        : ptr_(memory::Alloc(place, size)), size_(size), place_(place) {
      PADDLE_ENFORCE_GT(size_, 0, "Allocated size must be larger than 0");
      PADDLE_ENFORCE_NOT_NULL(ptr_, "Insufficient %s memory to allocation.",
                              (platform::is_cpu_place(place_) ? "CPU" : "GPU"));
    }

    ~InternalHolder() { memory::Free(place_, ptr_); }

    void *ptr_;    // Ensure ptr_ is not nullptr
    size_t size_;  // Ensure that size_ > 0
    platform::Place place_;
  };

 public:
#ifdef PADDLE_WITH_CUDA
  enum CopyType { kNone, kSync, kASync };
#else
  enum CopyType { kNone, kSync };
#endif

  bool IsInitialized() const { return holder_ != nullptr; }

  inline void Resize(const DDim &dims) { dims_ = dims; }

  inline void SetSizeOfElement(size_t size_of_element) {
    size_of_element_ = size_of_element;
  }

  inline size_t GetSizeOfElement() const { return size_of_element_; }

  void *GetMutable(const platform::Place &place, CopyType copy = kNone);

  inline const void *Get() const;

  inline const DDim &GetDims() const { return dims_; }

  inline size_t GetNumel() const { return static_cast<size_t>(product(dims_)); }

  inline void Clear();

  inline size_t GetMemorySize() const {
    PADDLE_ENFORCE_NE(size_of_element_, static_cast<size_t>(-1),
                      "Size of element has not set yet");
    return size_of_element_ * GetNumel();
  }

 private:
  std::shared_ptr<InternalHolder> holder_;
  DDim dims_;
  size_t size_of_element_{static_cast<size_t>(-1)};
  size_t offset_{0};
};

}  // namespace internal
}  // namespace framework
}  // namespace paddle
