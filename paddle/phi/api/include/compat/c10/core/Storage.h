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
class StorageHolderView;

// Check if two storages share the same underlying allocation
inline bool isSharedStorageAlias(const Storage& storage0,
                                 const Storage& storage1);

// Shared state for Storage handles. All Storage copies that refer to the
// same logical storage share a single StorageImpl, so mutations via
// set_data_ptr*(), set_nbytes(), or mutable_data_ptr() are visible to all
// handles — matching the observable contract of c10::StorageImpl in PyTorch.
struct StorageImpl {
  std::shared_ptr<phi::Allocation> data_allocation_;
  phi::Allocator* allocator_ = nullptr;
  size_t nbytes_ = 0;
  bool resizable_ = false;
  phi::Place place_;
  // DataPtr is stored directly (not in a shared_ptr) because all Storage
  // copies already share this StorageImpl, so one level of indirection
  // suffices to propagate mutations.
  DataPtr data_ptr_;
  std::weak_ptr<StorageHolderView> tensor_holder_;
};

class StorageHolderView final : public phi::Allocation {
 public:
  explicit StorageHolderView(std::shared_ptr<StorageImpl> impl)
      : impl_(std::move(impl)) {}

  std::shared_ptr<StorageImpl> get_impl() const { return impl_; }

  void* ptr() const noexcept override {
    if (!impl_) {
      return nullptr;
    }
    if (impl_->data_allocation_) {
      return impl_->data_allocation_->ptr();
    }
    return impl_->data_ptr_.get();
  }

  size_t size() const noexcept override { return impl_ ? impl_->nbytes_ : 0; }

  const Place& place() const noexcept override {
    return impl_ ? impl_->place_ : place_;
  }

 private:
  std::shared_ptr<StorageImpl> impl_;
  Place place_;
};

struct Storage {
 public:
  // Tag types for constructor disambiguation (LibTorch compatible)
  struct use_byte_size_t {};
  struct unsafe_borrow_t {
    unsafe_borrow_t() = default;
  };

  // Default constructor: empty storage, no allocation, no data.
  Storage() : impl_(std::make_shared<StorageImpl>()) {}

  // Copy constructor: shares the StorageImpl so that mutations made through
  // either handle are visible through the other.
  Storage(const Storage& other) : impl_(other.impl_) {}

  // Copy assignment operator
  Storage& operator=(const Storage& other) {
    if (this != &other) {
      impl_ = other.impl_;
    }
    return *this;
  }

  // Move constructor: transfers ownership of the StorageImpl.
  // The moved-from object is left in an unspecified (but destructible) state.
  Storage(Storage&& other) noexcept : impl_(std::move(other.impl_)) {}

  // Move assignment operator
  Storage& operator=(Storage&& other) noexcept {
    if (this != &other) {
      impl_ = std::move(other.impl_);
    }
    return *this;
  }

  // Constructor with allocation and optional storage properties
  Storage(std::shared_ptr<phi::Allocation> alloc,
          std::unique_ptr<phi::StorageProperties> props = nullptr) {
    impl_ = std::make_shared<StorageImpl>();
    if (alloc) {
      syncFromAllocation(std::move(alloc));
    }
  }

  // Constructor with size and allocator (LibTorch compatible)
  explicit Storage(size_t size_bytes, phi::Allocator* allocator = nullptr) {
    impl_ = std::make_shared<StorageImpl>();
    if (allocator) {
      auto alloc =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
      impl_->allocator_ = allocator;
      syncFromAllocation(std::move(alloc));
      impl_->nbytes_ = size_bytes;
    }
  }

  // LibTorch compatible constructor with use_byte_size_t tag
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    impl_ = std::make_shared<StorageImpl>();
    impl_->allocator_ = allocator;
    impl_->nbytes_ = size_bytes;
    impl_->resizable_ = resizable;
    if (allocator) {
      auto alloc =
          std::shared_ptr<phi::Allocation>(allocator->Allocate(size_bytes));
      syncFromAllocation(std::move(alloc));
      impl_->nbytes_ = size_bytes;
    }
  }

  // LibTorch compatible constructor with pre-allocated phi::Allocation
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          std::shared_ptr<phi::Allocation> alloc,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    impl_ = std::make_shared<StorageImpl>();
    impl_->allocator_ = allocator;
    impl_->nbytes_ = size_bytes;
    impl_->resizable_ = resizable;
    syncFromAllocation(std::move(alloc));
    impl_->nbytes_ = size_bytes;
  }

  // LibTorch compatible constructor with pre-allocated DataPtr
  Storage(use_byte_size_t /*use_byte_size*/,
          size_t size_bytes,
          DataPtr data_ptr,
          phi::Allocator* allocator = nullptr,
          bool resizable = false) {
    impl_ = std::make_shared<StorageImpl>();
    impl_->allocator_ = allocator;
    impl_->nbytes_ = size_bytes;
    impl_->resizable_ = resizable;
    syncFromDataPtr(std::move(data_ptr), size_bytes);
  }

 protected:
  // Unsafe borrow constructor (for MaybeOwnedTraits): shares the StorageImpl.
  // With the shared-impl design this is equivalent to a regular copy, but the
  // tag distinguishes "borrow" intent from ordinary copies at call sites.
  explicit Storage(unsafe_borrow_t, const Storage& rhs) : impl_(rhs.impl_) {}

  // Forward declare template and make specialization a friend
  template <typename T>
  friend struct MaybeOwnedTraits;

 public:
  // Construct from a pre-existing shared StorageImpl (used by the global
  // per-tensor storage registry to reuse an existing StorageImpl).
  explicit Storage(std::shared_ptr<StorageImpl> impl)
      : impl_(std::move(impl)) {}

  // Returns the underlying shared StorageImpl (used by the global per-tensor
  // storage registry).
  std::shared_ptr<StorageImpl> get_impl() const { return impl_; }

  static Storage createTensorStorage(
      const std::shared_ptr<phi::Allocation>& holder) {
    if (!holder) {
      return Storage();
    }
    if (auto storage_holder =
            std::dynamic_pointer_cast<StorageHolderView>(holder)) {
      return Storage(storage_holder->get_impl());
    }
    auto impl = std::make_shared<StorageImpl>();
    Storage storage(std::move(impl));
    storage.syncFromAllocation(holder);
    storage.ensureTensorHolder();
    return storage;
  }

  std::shared_ptr<phi::Allocation> ensureTensorHolder() const {
    if (!impl_) {
      return nullptr;
    }
    auto holder = impl_->tensor_holder_.lock();
    if (!holder) {
      holder = std::make_shared<StorageHolderView>(impl_);
      impl_->tensor_holder_ = holder;
    }
    return holder;
  }

  // Check if storage is valid (has allocation or data)
  bool valid() const {
    return impl_ && (static_cast<bool>(impl_->data_allocation_) ||
                     static_cast<bool>(impl_->data_ptr_));
  }

  // Boolean conversion operator (LibTorch compatible)
  explicit operator bool() const { return valid(); }

  // Get the number of bytes in the storage
  size_t nbytes() const { return impl_ ? impl_->nbytes_ : 0; }

  // Set the number of bytes.
  // For resizable storage with an allocator, reallocates; otherwise updates
  // the byte count directly so the change is visible to all copies.
  void set_nbytes(size_t size_bytes) {
    if (!impl_) return;
    if (impl_->resizable_ && impl_->allocator_) {
      syncFromAllocation(std::shared_ptr<phi::Allocation>(
          impl_->allocator_->Allocate(size_bytes)));
      impl_->nbytes_ = size_bytes;
    } else {
      impl_->nbytes_ = size_bytes;
    }
  }

  // Check if storage is resizable
  bool resizable() const { return impl_ ? impl_->resizable_ : false; }

  // Get mutable data pointer
  void* mutable_data() const {
    if (!impl_) {
      return nullptr;
    }
    if (impl_->data_allocation_) {
      return impl_->data_allocation_->ptr();
    }
    return impl_->data_ptr_.get();
  }

  // Get const data pointer
  const void* data() const {
    if (!impl_) {
      return nullptr;
    }
    if (impl_->data_allocation_) {
      return impl_->data_allocation_->ptr();
    }
    return impl_->data_ptr_.get();
  }

  // Get a const reference to the underlying DataPtr (LibTorch compatible)
  const DataPtr& data_ptr() const { return impl_->data_ptr_; }

  // Get a mutable reference to the underlying DataPtr (LibTorch compatible).
  // Because all Storage copies share the same StorageImpl, this reference
  // reflects and propagates changes to all handles — matching PyTorch's
  // StorageImpl semantics where mutable_data_ptr() returns the member directly.
  DataPtr& mutable_data_ptr() const { return impl_->data_ptr_; }

  // Get the underlying phi::Allocation (Paddle-specific)
  std::shared_ptr<phi::Allocation> allocation() const {
    return impl_ ? impl_->data_allocation_ : nullptr;
  }

  // Get the allocator
  phi::Allocator* allocator() const {
    return impl_ ? impl_->allocator_ : nullptr;
  }

  // Get the device/place type
  phi::AllocationType device_type() const {
    if (!impl_) return phi::AllocationType::CPU;
    if (impl_->data_allocation_)
      return impl_->data_allocation_->place().GetType();
    if (impl_->data_ptr_)
      return c10::DeviceTypeToPhi(impl_->data_ptr_.device().type());
    return phi::AllocationType::CPU;
  }

  // Get the device/place
  phi::Place device() const {
    if (!impl_) return phi::Place();
    if (impl_->data_allocation_) return impl_->data_allocation_->place();
    return impl_->place_;
  }

  // Returns the number of c10::Storage handles currently sharing this
  // StorageImpl (i.e. impl_.use_count()), matching PyTorch's
  // c10::Storage::use_count() semantics.  Returns 0 for empty / invalid
  // storage (neither allocation nor data_ptr set).
  size_t use_count() const {
    if (!valid()) return 0;
    size_t count = impl_.use_count();
    if (!impl_->tensor_holder_.expired() && count > 0) {
      --count;
    }
    return count;
  }

  // Check if this storage is unique (use_count == 1)
  bool unique() const { return use_count() == 1; }

  // Check if this storage is an alias of another
  bool is_alias_of(const Storage& other) const {
    if (!valid() || !other.valid()) {
      return false;
    }
    // Fast path: same StorageImpl (e.g. two copies of the same Storage handle)
    if (impl_ == other.impl_) return true;
    return isSharedStorageAlias(*this, other);
  }

  // Set data pointer (swap and return old) - LibTorch compatible DataPtr
  // version. Clears allocation-backed state since the new DataPtr manages its
  // own lifecycle. The change is propagated to all Storage copies that share
  // this StorageImpl. Use set_data_ptr(shared_ptr<phi::Allocation>) for
  // Paddle paths.
  DataPtr set_data_ptr(DataPtr&& new_data_ptr) {
    DataPtr old = std::move(impl_->data_ptr_);
    syncFromDataPtr(std::move(new_data_ptr), impl_->nbytes_);
    return old;
  }
  DataPtr set_data_ptr(std::nullptr_t) = delete;

  // Set data pointer (no swap) - LibTorch compatible DataPtr version.
  // Propagated to all Storage copies that share this StorageImpl.
  void set_data_ptr_noswap(DataPtr&& new_data_ptr) {
    syncFromDataPtr(std::move(new_data_ptr), impl_->nbytes_);
  }
  void set_data_ptr_noswap(std::nullptr_t) = delete;

  // Set data pointer - Paddle-specific shared_ptr<phi::Allocation> version.
  // Propagated to all Storage copies that share this StorageImpl.
  std::shared_ptr<phi::Allocation> set_data_ptr(
      std::shared_ptr<phi::Allocation> new_alloc) {
    std::shared_ptr<phi::Allocation> old_alloc =
        std::move(impl_->data_allocation_);
    syncFromAllocation(std::move(new_alloc));
    return old_alloc;
  }

  // Set data pointer (no swap) - Paddle-specific shared_ptr version.
  // Propagated to all Storage copies that share this StorageImpl.
  void set_data_ptr_noswap(std::shared_ptr<phi::Allocation> new_alloc) {
    syncFromAllocation(std::move(new_alloc));
  }

 private:
  // Shared implementation state. All Storage copies that were created by
  // copying this Storage share the same StorageImpl, so writes through any
  // handle are immediately visible through all other handles.
  std::shared_ptr<StorageImpl> impl_;

  // Update allocation-backed storage state. The tensor holder stays attached
  // to the shared StorageImpl, so tensors observe pointer changes without
  // extra registry state.
  void syncFromAllocation(std::shared_ptr<phi::Allocation> new_alloc) {
    impl_->data_allocation_ = std::move(new_alloc);
    if (impl_->data_allocation_) {
      impl_->nbytes_ = impl_->data_allocation_->size();
      impl_->place_ = impl_->data_allocation_->place();
    } else {
      impl_->nbytes_ = 0;
      impl_->place_ = phi::Place();
    }
    impl_->data_ptr_ = viewDataPtrFrom(impl_->data_allocation_);
  }

  void syncFromDataPtr(DataPtr&& new_data_ptr, size_t size_bytes) {
    impl_->data_allocation_ = nullptr;
    impl_->nbytes_ = size_bytes;
    impl_->place_ =
        new_data_ptr ? new_data_ptr.device()._PD_GetInner() : phi::Place();
    impl_->data_ptr_ = std::move(new_data_ptr);
  }

  // Create a non-owning DataPtr view of a phi::Allocation.
  // The allocation's lifetime is managed by impl_->data_allocation_.
  // No deleter is installed so the DataPtr holds only a raw pointer.
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
  c10::DeleterFnPtr deleter0 = storage0.data_ptr().get_deleter();
  c10::DeleterFnPtr deleter1 = storage1.data_ptr().get_deleter();
  if (deleter0 == nullptr || deleter1 == nullptr || deleter0 != deleter1) {
    return false;
  }
  void* context0 = storage0.data_ptr().get_context();
  void* context1 = storage1.data_ptr().get_context();
  return context0 != nullptr && context0 == context1;
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

  static void assignBorrow(borrow_type& lhs,  // NOLINT(runtime/references)
                           const borrow_type& rhs) {
    lhs = borrow_type(borrow_type::unsafe_borrow_t{}, rhs);
  }

  // NOLINTNEXTLINE(runtime/references)
  static void destroyBorrow(borrow_type& toDestroy) { toDestroy = Storage(); }

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

  static c10::Storage take(c10::Storage& x) {  // NOLINT(runtime/references)
    return std::move(x);
  }

  static pointer_type getImpl(repr_type& x) {  // NOLINT(runtime/references)
    return &x;
  }

  static const_pointer_type getImpl(const repr_type& x) { return &x; }
};

}  // namespace c10
