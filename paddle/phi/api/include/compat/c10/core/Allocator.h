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
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include "c10/util/Exception.h"
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

  // Release context ownership without calling deleter.
  // The caller takes responsibility for freeing via the deleter.
  // Used by raw_allocate() / raw_deallocate() API.
  void* release_context() {
    void* ctx = ctx_;
    ctx_ = nullptr;
    deleter_ = nullptr;
    ptr_ = nullptr;
    allocation_.reset();
    return ctx;
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

struct Allocator {
  virtual ~Allocator() = default;

  virtual DataPtr allocate(size_t n) = 0;

  // Clones an allocation that came from this allocator.
  //
  // To perform the copy, this function calls `copy_data`, which
  // must be implemented by derived classes.
  //
  // Note that this explicitly ignores any context that may have been
  // attached to the input data.
  //
  // Requires: input data was allocated by the same allocator.
  DataPtr clone(const void* data, std::size_t n) {
    auto new_data = allocate(n);
    copy_data(new_data.get(), data, n);
    return new_data;
  }

  // Checks if DataPtr has a simple context, not wrapped with any out of the
  // ordinary contexts.
  virtual bool is_simple_data_ptr(const DataPtr& data_ptr) const {
    return data_ptr.get_context() == nullptr ||
           data_ptr.get_context() == data_ptr.get();
  }

  // If this returns a non nullptr, it means that allocate()
  // is guaranteed to return a unique_ptr with this deleter attached;
  // it means the rawAllocate and rawDeallocate APIs are safe to use.
  // This function MUST always return the same BoundDeleter.
  virtual DeleterFnPtr raw_deleter() const { return nullptr; }
  void* raw_allocate(size_t n) {
    auto dptr = allocate(n);
    TORCH_CHECK(dptr.get() == dptr.get_context(),
                "raw_allocate: DataPtr context must equal data pointer");
    return dptr.release_context();
  }
  void raw_deallocate(void* ptr) {
    auto d = raw_deleter();
    TORCH_CHECK(d != nullptr, "raw_deallocate: deleter must not be null");
    d(ptr);
  }

  // Copies data from one allocation to another.
  // Pure virtual, so derived classes must define behavior.
  // Derived class implementation can simply call `default_copy_data`
  // to use `std::memcpy`.
  //
  // Requires: src and dest were allocated by this allocator
  // Requires: src and dest both have length >= count
  virtual void copy_data(void* dest,
                         const void* src,
                         std::size_t count) const = 0;

 protected:
  // Uses `std::memcpy` to copy data.
  // Child classes can use this as `copy_data` when an alternative copy
  // API is not needed.
  void default_copy_data(void* dest, const void* src, std::size_t count) const {
    std::memcpy(dest, src, count);
  }
};

}  // namespace c10

namespace at {
using DataPtr = c10::DataPtr;
}  // namespace at
