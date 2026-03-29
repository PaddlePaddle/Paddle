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

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/util/Exception.h>
#include <c10/util/UniqueVoidPtr.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"

namespace c10 {

// Deleter function pointer type (compatible with LibTorch)
using DeleterFnPtr = void (*)(void*);

using CaptureId_t = uint64_t;
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

// DataPtr class compatible with LibTorch's c10::DataPtr
// Wraps a pointer with associated device and deleter
class DataPtr {
 public:
  DataPtr() : device_(phi::CPUPlace()) {}

  DataPtr(void* data, Device device)
      : ptr_(data), device_(device._PD_GetInner()) {}

  DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
      : ptr_(data, ctx, ctx_deleter), device_(device._PD_GetInner()) {}

  // DataPtr is move-only, matching PyTorch's c10::DataPtr interface.
  DataPtr(const DataPtr&) = delete;
  DataPtr& operator=(const DataPtr&) = delete;
  DataPtr(DataPtr&&) = default;
  DataPtr& operator=(DataPtr&&) = default;

  void* operator->() const { return ptr_.get(); }

  bool unsafe_reset_data_and_ctx(void* new_data_and_ctx) {
    return ptr_.unsafe_reset_data_and_ctx(new_data_and_ctx);
  }

  void clear() { ptr_.clear(); }
  void* get() const { return ptr_.get(); }
  void* mutable_get() { return ptr_.get(); }
  void* get_context() const { return ptr_.get_context(); }
  void* release_context() { return ptr_.release_context(); }

  std::unique_ptr<void, DeleterFnPtr>&& move_context() {
    return ptr_.move_context();
  }

  operator bool() const { return static_cast<bool>(ptr_); }

  template <typename T>
  T* cast_context(DeleterFnPtr expected_deleter) const {
    return ptr_.cast_context<T>(expected_deleter);
  }

  DeleterFnPtr get_deleter() const { return ptr_.get_deleter(); }

  // Atomically replaces the deleter if it matches expected_deleter.
  // Returns true and installs new_deleter on match; does nothing and
  // returns false otherwise.
  [[nodiscard]] bool compare_exchange_deleter(DeleterFnPtr expected_deleter,
                                              DeleterFnPtr new_deleter) {
    return ptr_.compare_exchange_deleter(expected_deleter, new_deleter);
  }

  Device device() const { return Device(device_); }

  void unsafe_set_device(Device device) { device_ = device._PD_GetInner(); }

 private:
  c10::detail::UniqueVoidPtr ptr_;
  phi::Place device_;
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
    copy_data(new_data.mutable_get(), data, n);
    return new_data;
  }

  // Checks if DataPtr has a simple context, not wrapped with any out of the
  // ordinary contexts.
  virtual bool is_simple_data_ptr(const DataPtr& data_ptr) const {
    return data_ptr.get() == data_ptr.get_context();
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

struct InefficientStdFunctionContext {
  void* ptr_{nullptr};
  std::function<void(void*)> deleter_;

  InefficientStdFunctionContext(void* ptr, std::function<void(void*)> deleter)
      : ptr_(ptr), deleter_(std::move(deleter)) {}

  InefficientStdFunctionContext(const InefficientStdFunctionContext&) = delete;

  InefficientStdFunctionContext(InefficientStdFunctionContext&& rhs) noexcept
      : ptr_(std::exchange(rhs.ptr_, nullptr)),
        deleter_(std::move(rhs.deleter_)) {}

  InefficientStdFunctionContext& operator=(
      const InefficientStdFunctionContext&) = delete;

  InefficientStdFunctionContext& operator=(
      InefficientStdFunctionContext&& rhs) {
    this->~InefficientStdFunctionContext();
    ptr_ = std::exchange(rhs.ptr_, nullptr);
    deleter_ = std::move(rhs.deleter_);
    return *this;
  }

  ~InefficientStdFunctionContext() {
    if (deleter_) {
      deleter_(ptr_);
    }
  }

  static DataPtr makeDataPtr(void* ptr,
                             std::function<void(void*)> deleter,
                             Device device) {
    return DataPtr(ptr,
                   new InefficientStdFunctionContext(ptr, std::move(deleter)),
                   &deleteContext,
                   device);
  }

 private:
  static void deleteContext(void* ptr) {
    delete static_cast<InefficientStdFunctionContext*>(ptr);
  }
};

inline constexpr size_t kAllocatorRegistrySize =
    static_cast<size_t>(DeviceType::CUSTOM) + 1;

inline std::array<Allocator*, kAllocatorRegistrySize> g_allocator_array{};
inline std::array<uint8_t, kAllocatorRegistrySize> g_allocator_priority{};

inline size_t allocator_device_index(DeviceType t) {
  const size_t index = static_cast<size_t>(t);
  TORCH_CHECK(index < kAllocatorRegistrySize,
              "Allocator device type index out of range: ",
              index);
  return index;
}

inline void SetAllocator(DeviceType t, Allocator* alloc, uint8_t priority = 0) {
  const size_t index = allocator_device_index(t);
  if (priority >= g_allocator_priority[index]) {
    g_allocator_array[index] = alloc;
    g_allocator_priority[index] = priority;
  }
}

inline Allocator* GetAllocator(const DeviceType& t) {
  const size_t index = allocator_device_index(t);
  auto* alloc = g_allocator_array[index];
  TORCH_CHECK(alloc != nullptr, "Allocator for ", t, " is not set.");
  return alloc;
}

template <DeviceType t>
struct AllocatorRegisterer {
  explicit AllocatorRegisterer(Allocator* alloc) { SetAllocator(t, alloc); }
};

#define REGISTER_ALLOCATOR(t, f)                       \
  namespace {                                          \
  static c10::AllocatorRegisterer<t> g_allocator_d(f); \
  }

}  // namespace c10

namespace at {
using DataPtr = c10::DataPtr;
}  // namespace at
