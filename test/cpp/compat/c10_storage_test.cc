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

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

TEST(StorageTest, BasicStorageAPIs) {
  // Test basic Storage APIs through TensorBase
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  const c10::Storage& storage = tensor.storage();

  // Test valid()
  ASSERT_TRUE(storage.valid());

  // Test nbytes()
  size_t expected_nbytes = 2 * 3 * sizeof(float);
  ASSERT_EQ(storage.nbytes(), expected_nbytes);

  // Test data() and mutable_data()
  ASSERT_NE(storage.data(), nullptr);
  ASSERT_NE(storage.mutable_data(), nullptr);
  ASSERT_EQ(storage.data(), storage.mutable_data());

  // Test allocation()
  auto alloc = storage.allocation();
  ASSERT_NE(alloc, nullptr);
  ASSERT_EQ(alloc->size(), expected_nbytes);

  // Test unique() and use_count()
  // Note: In PaddlePaddle, DenseTensor holds a reference, Storage holds one,
  // and there may be additional internal references during tensor creation
  ASSERT_FALSE(storage.unique());
  ASSERT_EQ(storage.use_count(), 3);
}

TEST(StorageTest, StorageSharing) {
  // Test storage sharing between tensors
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1;  // Shared storage

  const c10::Storage& storage1 = tensor1.storage();
  const c10::Storage& storage2 = tensor2.storage();

  // Test that storages are the same
  ASSERT_EQ(storage1.allocation(), storage2.allocation());

  // Test use_count
  // Note: In PaddlePaddle, the count includes:
  // 1. DenseTensor's internal holder_
  // 2. storage1's allocation_
  // 3. storage2's allocation_
  // Total: 3
  ASSERT_EQ(storage1.use_count(), 3);
  ASSERT_EQ(storage2.use_count(), 3);

  // Test unique() is false
  ASSERT_FALSE(storage1.unique());
  ASSERT_FALSE(storage2.unique());
}

TEST(StorageTest, StorageOffsetAPI) {
  // Test storage_offset() API
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  // Test storage_offset() - should always return 0 for PaddlePaddle
  ASSERT_EQ(tensor.storage_offset(), 0);

  // Test sym_storage_offset() - should always return SymInt(0) for PaddlePaddle
  c10::SymInt sym_offset = tensor.sym_storage_offset();
  ASSERT_EQ(sym_offset, c10::SymInt(0));
}

TEST(StorageTest, IsAliasOfAPI) {
  // Test is_alias_of() API
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1;  // Shared storage, should be alias
  at::TensorBase tensor3 = at::ones({2, 3}, at::kFloat);  // Different storage

  // Test that tensor1 and tensor2 are aliases (share same storage)
  ASSERT_TRUE(tensor1.is_alias_of(tensor2));
  ASSERT_TRUE(tensor2.is_alias_of(tensor1));

  // Test that tensor1 and tensor3 are not aliases (different storage)
  ASSERT_FALSE(tensor1.is_alias_of(tensor3));
  ASSERT_FALSE(tensor3.is_alias_of(tensor1));

  // Test that tensor is alias of itself
  ASSERT_TRUE(tensor1.is_alias_of(tensor1));
}

TEST(StorageTest, BoolConversionOperator) {
  // Test operator bool() for Storage
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  const c10::Storage& storage = tensor.storage();

  // Valid storage should convert to true
  ASSERT_TRUE(static_cast<bool>(storage));

  // Default constructed storage should convert to false
  c10::Storage empty_storage;
  ASSERT_FALSE(static_cast<bool>(empty_storage));
  ASSERT_FALSE(empty_storage.valid());
}

TEST(StorageTest, ResizableAPI) {
  // Test resizable() API
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  const c10::Storage& storage = tensor.storage();

  // Default storage from tensor should not be resizable
  ASSERT_FALSE(storage.resizable());
}

TEST(StorageTest, DeviceAndDeviceTypeAPIs) {
  // Test device() and device_type() APIs
  at::TensorBase cpu_tensor = at::ones({2, 3}, at::kFloat);
  const c10::Storage& cpu_storage = cpu_tensor.storage();

  // Test device_type() returns CPU
  ASSERT_EQ(cpu_storage.device_type(), phi::AllocationType::CPU);

  // Test device() returns valid place
  phi::Place place = cpu_storage.device();
  ASSERT_EQ(place.GetType(), phi::AllocationType::CPU);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (at::cuda::is_available()) {
    at::TensorBase cuda_tensor = at::ones(
        {2, 3}, c10::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
    const c10::Storage& cuda_storage = cuda_tensor.storage();

    // Test device_type() returns CUDA/GPU
    ASSERT_EQ(cuda_storage.device_type(), phi::AllocationType::GPU);

    // Test device() returns CUDA place
    phi::Place cuda_place = cuda_storage.device();
    ASSERT_EQ(cuda_place.GetType(), phi::AllocationType::GPU);
  }
#endif
}

TEST(StorageTest, AllocatorAPI) {
  // Test allocator() API
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  const c10::Storage& storage = tensor.storage();

  // Allocator may be nullptr for storage obtained from tensor
  // This is expected behavior in the compatibility layer
  phi::Allocator* allocator = storage.allocator();
  // Note: allocator can be nullptr, this is just to verify the API works
  (void)allocator;
}

TEST(StorageTest, UnsafeAllocationAPIs) {
  // Test unsafeGetAllocation() API
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage storage = tensor.storage();

  // Test unsafeGetAllocation()
  phi::Allocation* alloc_ptr = storage.unsafeGetAllocation();
  ASSERT_NE(alloc_ptr, nullptr);
  ASSERT_EQ(alloc_ptr->size(), 2 * 3 * sizeof(float));

  // Test that the pointer matches the data pointer
  ASSERT_EQ(alloc_ptr->ptr(), storage.data());
}

TEST(StorageTest, SetDataPtrAPIs) {
  // Test set_data_ptr() and set_data_ptr_noswap() APIs
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = at::ones({4, 5}, at::kFloat);

  c10::Storage storage1 = tensor1.storage();
  c10::Storage storage2 = tensor2.storage();

  auto alloc1 = storage1.allocation();
  auto alloc2 = storage2.allocation();

  // Test set_data_ptr() - swaps and returns old
  auto old_alloc = storage1.set_data_ptr(alloc2);
  ASSERT_EQ(old_alloc, alloc1);
  ASSERT_EQ(storage1.allocation(), alloc2);

  // Test set_data_ptr_noswap()
  storage1.set_data_ptr_noswap(alloc1);
  ASSERT_EQ(storage1.allocation(), alloc1);
}

TEST(StorageTest, StorageCopyAndMove) {
  // Test copy and move constructors/operators
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage original = tensor.storage();

  // Test copy constructor
  c10::Storage copied(original);
  ASSERT_EQ(copied.allocation(), original.allocation());
  ASSERT_EQ(copied.nbytes(), original.nbytes());
  ASSERT_TRUE(copied.valid());

  // Test copy assignment
  c10::Storage copy_assigned;
  copy_assigned = original;
  ASSERT_EQ(copy_assigned.allocation(), original.allocation());

  // Test move constructor
  c10::Storage to_move = original;
  auto alloc_before_move = to_move.allocation();
  c10::Storage moved(std::move(to_move));
  ASSERT_EQ(moved.allocation(), alloc_before_move);
  ASSERT_TRUE(moved.valid());

  // Test move assignment
  c10::Storage to_move2 = original;
  alloc_before_move = to_move2.allocation();
  c10::Storage move_assigned;
  move_assigned = std::move(to_move2);
  ASSERT_EQ(move_assigned.allocation(), alloc_before_move);
}

TEST(StorageTest, DefaultConstructedStorage) {
  // Test default constructed storage
  c10::Storage storage;

  ASSERT_FALSE(storage.valid());
  ASSERT_FALSE(static_cast<bool>(storage));
  ASSERT_EQ(storage.nbytes(), 0);
  ASSERT_EQ(storage.data(), nullptr);
  ASSERT_EQ(storage.mutable_data(), nullptr);
  ASSERT_EQ(storage.allocation(), nullptr);
  ASSERT_EQ(storage.use_count(), 0);
  ASSERT_FALSE(storage.resizable());
  ASSERT_EQ(storage.allocator(), nullptr);
}

TEST(StorageTest, IsSharedStorageAliasFunction) {
  // Test isSharedStorageAlias() function
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1;                       // Shared storage
  at::TensorBase tensor3 = at::ones({2, 3}, at::kFloat);  // Different storage

  c10::Storage storage1 = tensor1.storage();
  c10::Storage storage2 = tensor2.storage();
  c10::Storage storage3 = tensor3.storage();

  // Same allocation should return true
  ASSERT_TRUE(c10::isSharedStorageAlias(storage1, storage2));
  ASSERT_TRUE(c10::isSharedStorageAlias(storage2, storage1));

  // Different allocations should return false
  ASSERT_FALSE(c10::isSharedStorageAlias(storage1, storage3));
  ASSERT_FALSE(c10::isSharedStorageAlias(storage3, storage1));

  // Empty storage should return false
  c10::Storage empty_storage;
  ASSERT_FALSE(c10::isSharedStorageAlias(storage1, empty_storage));
  ASSERT_FALSE(c10::isSharedStorageAlias(empty_storage, storage1));
  ASSERT_FALSE(c10::isSharedStorageAlias(empty_storage, empty_storage));
}

TEST(StorageTest, StorageIsAliasOfMethod) {
  // Test Storage::is_alias_of() method
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1;
  at::TensorBase tensor3 = at::ones({2, 3}, at::kFloat);

  c10::Storage storage1 = tensor1.storage();
  c10::Storage storage2 = tensor2.storage();
  c10::Storage storage3 = tensor3.storage();

  // Same underlying allocation
  ASSERT_TRUE(storage1.is_alias_of(storage2));
  ASSERT_TRUE(storage2.is_alias_of(storage1));

  // Different allocations
  ASSERT_FALSE(storage1.is_alias_of(storage3));

  // Self alias
  ASSERT_TRUE(storage1.is_alias_of(storage1));

  // Empty storage
  c10::Storage empty_storage;
  ASSERT_FALSE(storage1.is_alias_of(empty_storage));
  ASSERT_FALSE(empty_storage.is_alias_of(storage1));
}

TEST(StorageTest, MaybeOwnedTraitsSpecialization) {
  // Test MaybeOwnedTraits<c10::Storage> specialization
  using Traits = c10::MaybeOwnedTraits<c10::Storage>;

  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage original = tensor.storage();

  // Test createBorrow
  Traits::borrow_type borrowed = Traits::createBorrow(original);
  ASSERT_EQ(borrowed.allocation(), original.allocation());
  ASSERT_TRUE(borrowed.valid());

  // Test referenceFromBorrow
  const Traits::owned_type& ref = Traits::referenceFromBorrow(borrowed);
  ASSERT_EQ(ref.allocation(), original.allocation());

  // Test pointerFromBorrow
  const Traits::owned_type* ptr = Traits::pointerFromBorrow(borrowed);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->allocation(), original.allocation());

  // Test debugBorrowIsValid
  ASSERT_TRUE(Traits::debugBorrowIsValid(borrowed));

  // Test assignBorrow
  c10::Storage another_borrow;
  Traits::assignBorrow(&another_borrow, borrowed);
  ASSERT_EQ(another_borrow.allocation(), original.allocation());

  // Test destroyBorrow
  Traits::destroyBorrow(&borrowed);
  ASSERT_FALSE(borrowed.valid());
}

TEST(StorageTest, ExclusivelyOwnedTraitsSpecialization) {
  // Test ExclusivelyOwnedTraits<c10::Storage> specialization
  using Traits = c10::ExclusivelyOwnedTraits<c10::Storage>;

  // Test nullRepr
  Traits::repr_type null_repr = Traits::nullRepr();
  ASSERT_FALSE(null_repr.valid());

  // Test createInPlace with default constructor
  Traits::repr_type created = Traits::createInPlace();
  ASSERT_FALSE(created.valid());  // Default constructed

  // Test moveToRepr
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage original = tensor.storage();
  auto alloc = original.allocation();
  Traits::repr_type moved = Traits::moveToRepr(std::move(original));
  ASSERT_EQ(moved.allocation(), alloc);

  // Test take
  c10::Storage to_take = tensor.storage();
  alloc = to_take.allocation();
  c10::Storage taken = Traits::take(&to_take);
  ASSERT_EQ(taken.allocation(), alloc);

  // Test getImpl (mutable)
  Traits::pointer_type ptr = Traits::getImpl(&taken);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->allocation(), alloc);

  // Test getImpl (const)
  const c10::Storage& const_taken = taken;
  Traits::const_pointer_type const_ptr = Traits::getImpl(const_taken);
  ASSERT_NE(const_ptr, nullptr);
  ASSERT_EQ(const_ptr->allocation(), alloc);
}
