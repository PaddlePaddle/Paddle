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
  Storage() : data_ptr_(std::make_shared<DataPtr>()) {}

  // Copy constructor: shares both allocation (increments refcount) and the
  // DataPtr owner so that any context/deleter is preserved across copies.
  Storage(const Storage& other)
      : allocation_(other.allocation_),
        allocator_(other.allocator_),
        nbytes_(other.nbytes_),
        resizable_(other.resizable_),
        data_ptr_(other.data_ptr_) {}

  // Copy assignment operator
  Storage& operator=(const Storage& other) {
    if (this != &other) {
      allocation_ = other.allocation_;
      allocator_ = other.allocator_;
      nbytes_ = other.nbytes_;
      resizable_ = other.resizable_;
      data_ptr_ = other.data_ptr_;
    }
    return *this;
  }

  // Move constructor
  Storage(Storage&& other) noexcept
      : allocation_(std::move(other.allocation_)),
        allocator_(other.allocator_),
        nbytes_(other.nbytes_),
        resizable_(other.resizable_),
        data_ptr_(std::move(other.data_ptr_)) {
    other.allocator_ = nullptr;
    other.nbytes_ = 0;
    other.resizable_ = false;
    other.data_ptr_ = std::make_shared<DataPtr>();
  }

  // Move assignment operator
  Storage& operator=(Storage&& other) noexcept {
    if (this != &other) {
      allocation_ = std::move(other.allocation_);
      allocator_ = other.allocator_;
      nbytes_ = other.nbytes_;
      resizable_ = other.resizable_;
      data_ptr_ = std::move(other.data_ptr_);
      other.allocator_ = nullptr;
      other.nbytes_ = 0;
      other.resizable_ = false;
      other.data_ptr_ = std::make_shared<DataPtr>();
    }
    return *this;
  }

  // Constructor with allocation and optional storage properties
  Storage(std::shared_ptr<phi::Allocation> alloc,
          std::unique_ptr<phi::StorageProperties> props = nullptr) {
    if (alloc) {
      nbytes_ = alloc->size();
      allocation_ = alloc;
      data_ptr_ = std::make_shared<DataPtr>(viewDataPtrFrom(std::move(alloc)));
    } else {
      data_ptr_ = std::make_shared<DataPtr>();
    }
  }

  // Constructor with size and allocator (LibTorch compatible)
  explicit Storage(size_t size_bytes, phi::Allocator* allocator = nullptr) {
    if (allocator) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
      allocator_ = allocator;
      nbytes_ = size_bytes;
      data_ptr_ = std::make_shared<DataPtr>(viewDataPtrFrom(allocation_));
    } else {
      data_ptr_ = std::make_shared<DataPtr>();
    }
  }

  // LibTorch compatible constructor with use_byte_size_t tag
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    allocator_ = allocator;
    nbytes_ = size_bytes;
    resizable_ = resizable;
    if (allocator) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
      data_ptr_ = std::make_shared<DataPtr>(viewDataPtrFrom(allocation_));
    } else {
      data_ptr_ = std::make_shared<DataPtr>();
    }
  }

  // LibTorch compatible constructor with pre-allocated phi::Allocation
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          std::shared_ptr<phi::Allocation> alloc,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    allocation_ = alloc;
    allocator_ = allocator;
    nbytes_ = size_bytes;
    resizable_ = resizable;
    data_ptr_ = std::make_shared<DataPtr>(viewDataPtrFrom(std::move(alloc)));
  }

  // LibTorch compatible constructor with pre-allocated DataPtr
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          DataPtr data_ptr,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    allocator_ = allocator;
    nbytes_ = size_bytes;
    resizable_ = resizable;
    data_ptr_ = std::make_shared<DataPtr>(std::move(data_ptr));
  }

 protected:
  // Unsafe borrow constructor (for MaybeOwnedTraits): shares DataPtr ownership
  // so context/deleter is preserved without transfer of ownership.
  explicit Storage(unsafe_borrow_t, const Storage& rhs) {
    allocation_ = rhs.allocation_;
    allocator_ = rhs.allocator_;
    nbytes_ = rhs.nbytes_;
    resizable_ = rhs.resizable_;
    data_ptr_ = rhs.data_ptr_;
  }

  // Forward declare template and make specialization a friend
  template <typename T>
  friend struct MaybeOwnedTraits;

 public:
  // Check if storage is valid (has allocation or data)
  bool valid() const {
    return static_cast<bool>(allocation_) ||
           (data_ptr_ && static_cast<bool>(*data_ptr_));
  }

  // Boolean conversion operator (LibTorch compatible)
  explicit operator bool() const { return valid(); }

  // Get the number of bytes in the storage
  size_t nbytes() const { return nbytes_; }

  // Set the number of bytes (for resizable storage)
  void set_nbytes(size_t size_bytes) {
    if (resizable_ && allocator_) {
      allocation_ =
          std::shared_ptr<phi::Allocation>(allocator_->Allocate(size_bytes));
      data_ptr_ = std::make_shared<DataPtr>(viewDataPtrFrom(allocation_));
      nbytes_ = size_bytes;
    }
  }

  // Check if storage is resizable
  bool resizable() const { return resizable_; }

  // Get mutable data pointer
  void* mutable_data() const { return data_ptr_->get(); }

  // Get const data pointer
  const void* data() const { return data_ptr_->get(); }

  // Get a const reference to the underlying DataPtr (LibTorch compatible)
  const DataPtr& data_ptr() const { return *data_ptr_; }

  // Get a mutable reference to the underlying DataPtr (LibTorch compatible).
  // Implements copy-on-write: if data_ptr_ is shared across copies, this
  // storage detaches by creating a non-owning view so that mutations here do
  // not affect other Storage copies that share the same DataPtr.  The owning
  // DataPtr (and its context/deleter) remains held by the other copies,
  // keeping the allocation alive.
  DataPtr& mutable_data_ptr() const {
    if (data_ptr_.use_count() > 1) {
      void* raw = data_ptr_->get();
      c10::Device dev = data_ptr_->device();
      const_cast<Storage*>(this)->data_ptr_ =
          std::make_shared<DataPtr>(DataPtr(raw, dev));
    }
    return *data_ptr_;
  }

  // Get the underlying phi::Allocation (Paddle-specific)
  std::shared_ptr<phi::Allocation> allocation() const { return allocation_; }

  // Get the allocator
  phi::Allocator* allocator() const { return allocator_; }

  // Get the device/place type
  phi::AllocationType device_type() const {
    if (allocation_) return allocation_->place().GetType();
    return phi::AllocationType::CPU;
  }

  // Get the device/place
  phi::Place device() const {
    if (allocation_) return allocation_->place();
    return phi::Place();
  }

  // Check if this storage is unique (use_count == 1)
  bool unique() const { return allocation_.use_count() == 1; }

  // Get the reference count
  size_t use_count() const { return allocation_.use_count(); }

  // Check if this storage is an alias of another
  bool is_alias_of(const Storage& other) const {
    if (!valid() || !other.valid()) {
      return false;
    }
    return allocation_ == other.allocation_ ||
           isSharedStorageAlias(*this, other);
  }

  // Set data pointer (swap and return old) - LibTorch compatible DataPtr
  // version. Clears allocation_ since the new DataPtr manages its own
  // lifecycle. Uses in-place replacement so that all Storage copies sharing
  // this data_ptr_ see the new DataPtr rather than a moved-from empty state.
  // Use set_data_ptr(shared_ptr<phi::Allocation>) for Paddle paths.
  DataPtr set_data_ptr(DataPtr&& new_data_ptr) {
    DataPtr old = std::move(*data_ptr_);
    *data_ptr_ = std::move(new_data_ptr);
    allocation_ = nullptr;
    return old;
  }

  // Set data pointer (no swap) - LibTorch compatible DataPtr version
  void set_data_ptr_noswap(DataPtr&& new_data_ptr) {
    *data_ptr_ = std::move(new_data_ptr);
    allocation_ = nullptr;
  }

  // Set data pointer - Paddle-specific shared_ptr<phi::Allocation> version
  std::shared_ptr<phi::Allocation> set_data_ptr(
      std::shared_ptr<phi::Allocation> new_alloc) {
    std::shared_ptr<phi::Allocation> old_alloc = std::move(allocation_);
    allocation_ = new_alloc;
    if (allocation_) nbytes_ = allocation_->size();
    data_ptr_ =
        std::make_shared<DataPtr>(viewDataPtrFrom(std::move(new_alloc)));
    return old_alloc;
  }

  // Set data pointer (no swap) - Paddle-specific shared_ptr version
  void set_data_ptr_noswap(std::shared_ptr<phi::Allocation> new_alloc) {
    allocation_ = new_alloc;
    if (allocation_) nbytes_ = allocation_->size();
    data_ptr_ =
        std::make_shared<DataPtr>(viewDataPtrFrom(std::move(new_alloc)));
  }

 private:
  // Member declaration order matters for initializer-list initialization.
  // allocation_ must come before data_ptr_ so that viewDataPtrFrom can
  // use allocation_ in constructors that initialize data_ptr_ from it.
  std::shared_ptr<phi::Allocation> allocation_;
  phi::Allocator* allocator_ = nullptr;
  size_t nbytes_ = 0;
  bool resizable_ = false;
  // Shared pointer to DataPtr — shared across Storage copies to preserve
  // context/deleter on the external-DataPtr path.  For allocation-backed
  // paths the pointed-to DataPtr is a non-owning view (no extra refcount on
  // allocation_), so use_count() remains accurate.
  std::shared_ptr<DataPtr> data_ptr_;

  // Create a non-owning DataPtr view of a phi::Allocation.
  // The allocation's lifetime is managed separately by allocation_.
  // This does NOT increment the shared_ptr refcount, so use_count() stays
  // accurate.
  static DataPtr viewDataPtrFrom(
      const std::shared_ptr<phi::Allocation>& alloc) {
    if (!alloc) return DataPtr();
    return DataPtr(alloc->ptr(), c10::Device(alloc->place()));
  }
};

// Implementation of isSharedStorageAlias
inline bool isSharedStorageAlias(const Storage& storage0,
                                 const Storage& storage1) {
  if (!storage0.valid() || !storage1.valid()) {
    return false;
  }
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

  return !(end0 <= start1 || end1 <= start0);
}

// Template specialization for MaybeOwnedTraits<c10::Storage>
template <typename T>
struct MaybeOwnedTraits;

template <>
struct MaybeOwnedTraits<c10::Storage> {
  using owned_type = c10::Storage;
  using borrow_type = c10::Storage;

  static borrow_type createBorrow(const owned_type& from) {
    return borrow_type(borrow_type::unsafe_borrow_t{}, from);
  }

  static void assignBorrow(borrow_type* lhs, const borrow_type& rhs) {
    *lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  static void destroyBorrow(borrow_type* toDestroy) { *toDestroy = Storage(); }

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return &borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& /*borrow*/) { return true; }
};

// Template specialization for ExclusivelyOwnedTraits<c10::Storage>
template <typename T>
struct ExclusivelyOwnedTraits;

template <>
struct ExclusivelyOwnedTraits<c10::Storage> {
  using repr_type = c10::Storage;
  using pointer_type = c10::Storage*;
  using const_pointer_type = const c10::Storage*;

  static repr_type nullRepr() { return c10::Storage(); }

  template <class... Args>
  static repr_type createInPlace(Args&&... args) {
    return c10::Storage(std::forward<Args>(args)...);
  }

  static repr_type moveToRepr(c10::Storage&& x) { return std::move(x); }

  static c10::Storage take(c10::Storage* x) { return std::move(*x); }

  static pointer_type getImpl(repr_type* x) { return x; }

  static const_pointer_type getImpl(const repr_type& x) { return &x; }
};

}  // namespace c10
