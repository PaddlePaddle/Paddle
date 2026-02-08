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

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"

namespace c10 {

// Deleter function pointer type (compatible with LibTorch)
using DeleterFnPtr = void (*)(void*);

// DataPtr class compatible with LibTorch's c10::DataPtr
// Wraps a pointer with associated device and deleter
class DataPtr {
 public:
  DataPtr() : ptr_(nullptr), device_(phi::CPUPlace()) {}

  explicit DataPtr(void* data, phi::Place device = phi::CPUPlace())
      : ptr_(data), device_(device) {}

  DataPtr(void* data,
          void* ctx,
          DeleterFnPtr ctx_deleter,
          phi::Place device = phi::CPUPlace())
      : ptr_(data), ctx_(ctx), deleter_(ctx_deleter), device_(device) {}

  // Construct from phi::Allocation
  explicit DataPtr(const std::shared_ptr<phi::Allocation>& alloc)
      : ptr_(alloc ? alloc->ptr() : nullptr),
        device_(alloc ? alloc->place() : phi::CPUPlace()),
        allocation_(alloc) {}

  DataPtr(const DataPtr&) = default;
  DataPtr& operator=(const DataPtr&) = default;
  DataPtr(DataPtr&&) = default;
  DataPtr& operator=(DataPtr&&) = default;

  void* get() const { return ptr_; }

  void* operator->() const { return ptr_; }

  explicit operator bool() const { return ptr_ != nullptr; }

  phi::Place device() const { return device_; }

  DeleterFnPtr get_deleter() const { return deleter_; }

  void* get_context() const { return ctx_; }

  void clear() {
    ptr_ = nullptr;
    ctx_ = nullptr;
    deleter_ = nullptr;
    allocation_.reset();
  }

  // Get the underlying allocation (if available)
  std::shared_ptr<phi::Allocation> allocation() const { return allocation_; }

 private:
  void* ptr_ = nullptr;
  void* ctx_ = nullptr;
  DeleterFnPtr deleter_ = nullptr;
  phi::Place device_;
  std::shared_ptr<phi::Allocation> allocation_;
};

inline bool operator==(const DataPtr& dp, std::nullptr_t) noexcept {
  return !dp;
}

inline bool operator==(std::nullptr_t, const DataPtr& dp) noexcept {
  return !dp;
}

inline bool operator!=(const DataPtr& dp, std::nullptr_t) noexcept {
  return static_cast<bool>(dp);
}

inline bool operator!=(std::nullptr_t, const DataPtr& dp) noexcept {
  return static_cast<bool>(dp);
}

}  // namespace c10

namespace at {
using DataPtr = c10::DataPtr;
}  // namespace at
