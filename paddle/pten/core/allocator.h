/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <cstdint>
#include "paddle/fluid/platform/place.h"

namespace pten {

/// \brief Encapsulates strategies for access/addressing, allocation/
/// deallocation and construction/destruction of objects.
class RawAllocator {
 public:
  using Place = paddle::platform::Place;

  /// \brief Default destructor.
  virtual ~RawAllocator() = default;

  /// \brief Allocates storage suitable for an array object of n bytes
  /// and creates the array, but does not construct array elements.
  /// May throw exceptions.
  /// \param bytes_size The number of bytes to allocate.
  /// \return The first address allocated.
  virtual void* Allocate(size_t bytes_size) = 0;

  /// \brief Deallocates storage pointed to ptr, which must be a value
  /// returned by a previous call to allocate that has not been
  /// invalidated by an intervening call to deallocate. The bytes_size
  /// must match the value previously passed to allocate.
  /// \param ptr The first address to deallocate.
  /// \param bytes_size The number of bytes to deallocate.
  virtual void Deallocate(void* ptr, size_t bytes_size) = 0;

  /// \brief Get the place value of the allocator and the allocation.
  /// \return The place value of the allocator and the allocation.
  virtual const Place& place() const = 0;
};

/// \brief Fancy pointer with context. The use of this data type
/// is to be compatible with allocators from different frameworks
/// without significant performance loss. This class does not
/// support being inherited.
class Allocation final {
 public:
  using Place = paddle::platform::Place;
  using DeleterFnPtr = void (*)(void*);

  Allocation() = default;
  Allocation(Allocation&&) = default;
  Allocation& operator=(Allocation&&) = default;

  Allocation(void* data, const Place& place) : data_(data), place_(place) {}

  Allocation(void* data,
             void* ctx,
             DeleterFnPtr ctx_deleter,
             const Place& place)
      : data_(data), ctx_(ctx, ctx_deleter), place_(place) {}

  void* operator->() const noexcept { return data_; }
  operator bool() const noexcept { return data_ || ctx_.Get(); }
  const Place& place() const noexcept { return place_; }

  void Clear() {
    ctx_.Clear();
    data_ = nullptr;
  }

  /// \brief Statically cast the void pointer of the context object to
  /// the primitive type. Conversion of any pointer to void* and back
  /// to pointer to the original cv type preserves its original value.
  /// \param T The primitive type name of the context pointer.
  /// \param expected_deleter The destructor passed in to enhance type
  /// safety checking.
  template <typename T>
  T* CastContext(DeleterFnPtr expected_deleter) const noexcept {
    if (ctx_.deleter() != expected_deleter) {
      return nullptr;
    }
    return static_cast<T*>(ctx_.Get());
  }

 public:
  class Context {
   public:
    Context() = default;
    Context(void* ctx, DeleterFnPtr deleter) noexcept : ctx_(ctx),
                                                        deleter_(deleter) {}
    Context(Context&& other) noexcept {
      // Exchange them explicitly to avoid moving is equivalent
      // to copying.
      swap(*this, other);
    }
    Context& operator=(Context&& other) noexcept {
      swap(*this, other);
      return *this;
    }
    ~Context() { Clear(); }
    void Clear() {
      if (deleter_) {
        deleter_(ctx_);
      }
      ctx_ = nullptr;
      deleter_ = nullptr;
    }
    void* Get() const noexcept { return ctx_; }
    DeleterFnPtr deleter() const noexcept { return deleter_; }
    void* Release() noexcept {
      deleter_ = nullptr;
      return ctx_;
    }
    friend void swap(Context& a, Context& b) noexcept;

   private:
    void* ctx_{nullptr};
    DeleterFnPtr deleter_{nullptr};
  };

 private:
  void* data_{nullptr};
  Context ctx_;
  // TODO(Shixiaowei02): Enum needs to be used instead to reduce
  // the construction overhead by more than 50%.
  Place place_;
};

inline void swap(Allocation::Context& a, Allocation::Context& b) noexcept {
  ::std::swap(a.ctx_, b.ctx_);
  ::std::swap(a.deleter_, b.deleter_);
}

/// \brief Context compatible allocator interface. This allocator is
/// mainly used for general data structures such as Tensor. The raw
/// allocator is more universal and efficient.
class Allocator {
 public:
  virtual ~Allocator() = default;
  virtual Allocation Allocate(size_t bytes_size) = 0;
};

inline Allocation Allocate(const std::shared_ptr<Allocator>& a, size_t n) {
  CHECK(a);
  return a->Allocate(n);
}

}  // namespace pten
