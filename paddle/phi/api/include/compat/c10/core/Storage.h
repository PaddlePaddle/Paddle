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

#include <memory>
#include <utility>

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/storage_properties.h"

#include "c10/core/Allocator.h"  // For DataPtr

namespace c10 {

struct Storage;

// Check if two storages share the same underlying allocation
inline bool isSharedStorageAlias(const Storage& storage0,
                                 const Storage& storage1);

struct Storage {
 public:
  // Tag types for constructor disambiguation (LibTorch compatible)
  struct use_byte_size_t {};
  struct unsafe_borrow_t {
    unsafe_borrow_t() = default;
  };

  // Default constructor
  Storage() = default;

  // Copy constructor
  Storage(const Storage& other)
      : allocation_(other.allocation_),
        allocator_(other.allocator_),
        resizable_(other.resizable_) {}

  // Copy assignment operator
  Storage& operator=(const Storage& other) {
    if (this != &other) {
      allocation_ = other.allocation_;
      allocator_ = other.allocator_;
      resizable_ = other.resizable_;
    }
    return *this;
  }

  // Move constructor
  Storage(Storage&& other) noexcept
      : allocation_(std::move(other.allocation_)),
        allocator_(other.allocator_),
        resizable_(other.resizable_) {
    other.allocator_ = nullptr;
    other.resizable_ = false;
  }

  // Move assignment operator
  Storage& operator=(Storage&& other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      allocator_ = other.allocator_;
      resizable_ = other.resizable_;
      other.allocator_ = nullptr;
      other.resizable_ = false;
    }
    return *this;
  }

  // Constructor with allocation and optional storage properties
  Storage(std::shared_ptr<phi::Allocation> alloc,
          std::unique_ptr<phi::StorageProperties> props = nullptr)
      : allocation_(std::move(alloc)) {}

  // Constructor with size and allocator (LibTorch compatible)
  explicit Storage(size_t size_bytes, phi::Allocator* allocator = nullptr) {
    if (allocator) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
      allocator_ = allocator;
    } else {
      allocation_ = nullptr;
      allocator_ = nullptr;
    }
  }

  // LibTorch compatible constructor with use_byte_size_t tag
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          phi::Allocator* allocator = nullptr,
          bool resizable = false)
      : allocator_(allocator), resizable_(resizable) {
    if (allocator) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
    } else {
      allocation_ = nullptr;
    }
  }

  // LibTorch compatible constructor with pre-allocated memory
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          std::shared_ptr<phi::Allocation> data_ptr,
          phi::Allocator* allocator = nullptr,
          bool resizable = false)
      : allocation_(std::move(data_ptr)),
        allocator_(allocator),
        resizable_(resizable) {}

 protected:
  // Unsafe borrow constructor (for MaybeOwnedTraits)
  explicit Storage(unsafe_borrow_t, const Storage& rhs)
      : allocation_(rhs.allocation_),
        allocator_(rhs.allocator_),
        resizable_(rhs.resizable_) {}

  // Forward declare template and make specialization a friend
  template <typename T>
  friend struct MaybeOwnedTraits;

 public:
  // Check if storage is valid (has allocation)
  bool valid() const { return allocation_ != nullptr; }

  // Boolean conversion operator (LibTorch compatible)
  explicit operator bool() const { return allocation_ != nullptr; }

  // Get the number of bytes in the storage
  size_t nbytes() const { return allocation_ ? allocation_->size() : 0; }

  // Set the number of bytes (for resizable storage)
  void set_nbytes(size_t size_bytes) {
    if (resizable_ && allocator_) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator_->Allocate(size_bytes));
    }
  }

  // Check if storage is resizable
  bool resizable() const { return resizable_; }

  // Get mutable data pointer
  void* mutable_data() const {
    return allocation_ ? allocation_->ptr() : nullptr;
  }

  // Get const data pointer
  const void* data() const {
    return allocation_ ? allocation_->ptr() : nullptr;
  }

  // Get the underlying allocation as DataPtr (LibTorch compatible: data_ptr())
  DataPtr data_ptr() const { return DataPtr(allocation_); }

  // Get the underlying allocation as mutable DataPtr reference
  DataPtr mutable_data_ptr() const { return DataPtr(allocation_); }

  // Get the underlying allocation
  std::shared_ptr<phi::Allocation> allocation() const { return allocation_; }

  // Get the allocator
  phi::Allocator* allocator() const { return allocator_; }

  // Get the device/place type
  phi::AllocationType device_type() const {
    return allocation_ ? allocation_->place().GetType()
                       : phi::AllocationType::CPU;
  }

  // Get the device/place
  phi::Place device() const {
    return allocation_ ? allocation_->place() : phi::Place();
  }

  // Check if this storage is unique (use_count == 1)
  bool unique() const { return allocation_.use_count() == 1; }

  // Get the reference count
  size_t use_count() const { return allocation_.use_count(); }

  // Check if this storage is an alias of another
  bool is_alias_of(const Storage& other) const {
    if (!allocation_ || !other.allocation_) {
      return false;
    }
    // Check if they share the same allocation or overlapping memory
    return allocation_ == other.allocation_ ||
           isSharedStorageAlias(*this, other);
  }

  // Unsafe release of the underlying allocation (for advanced usage)
  phi::Allocation* unsafeReleaseAllocation() {
    auto* ptr = allocation_.get();
    allocation_.reset();
    return ptr;
  }

  // Unsafe get of the underlying allocation pointer
  phi::Allocation* unsafeGetAllocation() const noexcept {
    return allocation_.get();
  }

  // Set data pointer (swap and return old) - accepts DataPtr
  DataPtr set_data_ptr(DataPtr&& new_data_ptr) {
    DataPtr old_data_ptr(allocation_);
    allocation_ = new_data_ptr.allocation();
    return old_data_ptr;
  }

  // Set data pointer (swap and return old) - accepts shared_ptr
  std::shared_ptr<phi::Allocation> set_data_ptr(
      std::shared_ptr<phi::Allocation> data_ptr) {
    std::swap(allocation_, data_ptr);
    return data_ptr;
  }

  // Set data pointer (no swap) - accepts DataPtr
  void set_data_ptr_noswap(DataPtr&& new_data_ptr) {
    allocation_ = new_data_ptr.allocation();
  }

  // Set data pointer (no swap) - accepts shared_ptr
  void set_data_ptr_noswap(std::shared_ptr<phi::Allocation> data_ptr) {
    allocation_ = std::move(data_ptr);
  }

 private:
  std::shared_ptr<phi::Allocation> allocation_;
  phi::Allocator* allocator_ = nullptr;
  bool resizable_ = false;
};

// Implementation of isSharedStorageAlias
inline bool isSharedStorageAlias(const Storage& storage0,
                                 const Storage& storage1) {
  if (!storage0.valid() || !storage1.valid()) {
    return false;
  }
  // Check if memory ranges overlap
  const void* ptr0 = storage0.data();
  const void* ptr1 = storage1.data();
  size_t size0 = storage0.nbytes();
  size_t size1 = storage1.nbytes();

  if (ptr0 == nullptr || ptr1 == nullptr || size0 == 0 || size1 == 0) {
    return false;
  }

  const char* start0 = static_cast<const char*>(ptr0);
  const char* end0 = start0 + size0;
  const char* start1 = static_cast<const char*>(ptr1);
  const char* end1 = start1 + size1;

  // Check for overlap
  return !(end0 <= start1 || end1 <= start0);
}

// Template specialization for MaybeOwnedTraits<c10::Storage>
// Provides safe borrowing semantics for Storage objects
template <typename T>
struct MaybeOwnedTraits;

template <>
struct MaybeOwnedTraits<c10::Storage> {
  using owned_type = c10::Storage;
  using borrow_type = c10::Storage;

  // Create a borrowed reference from an owned Storage
  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type(borrow_type::unsafe_borrow_t{}, from);
  }

  // Assign a borrowed reference (LibTorch compatible signature with pointer)
  static void assignBorrow(borrow_type* lhs, const borrow_type& rhs) {
    *lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  // Destroy a borrowed reference (release without deallocating)
  static void destroyBorrow(borrow_type* toDestroy) {
    *toDestroy = Storage();  // Reset to empty state
  }

  // Get a reference to the owned object from a borrow
  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  // Get a pointer to the owned object from a borrow
  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  // Debug check if borrow is valid
  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) { return true; }
};

// Template specialization for ExclusivelyOwnedTraits<c10::Storage>
// Provides exclusive ownership semantics for Storage objects
template <typename T>
struct ExclusivelyOwnedTraits;

template <>
struct ExclusivelyOwnedTraits<c10::Storage> {
  using repr_type = c10::Storage;
  using pointer_type = c10::Storage*;
  using const_pointer_type = const c10::Storage*;

  // Create a null/empty representation
  static repr_type nullRepr() { return c10::Storage(); }

  // Create a Storage in place with given arguments
  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return c10::Storage(std::forward<Args>(args)...);
  }

  // Move a Storage into the representation
  static repr_type moveToRepr(c10::Storage&& x) { return std::move(x); }

  // Take ownership from a Storage pointer (LibTorch compatible)
  static c10::Storage take(c10::Storage* x) { return std::move(*x); }

  // Get a pointer to the representation (mutable)
  static pointer_type getImpl(repr_type* x) { return x; }

  // Get a const pointer to the representation
  static const_pointer_type getImpl(const repr_type& x) { return &x; }
};

}  // namespace c10
