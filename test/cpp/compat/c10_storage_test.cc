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
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/EmptyTensor.h>
#include <ATen/native/cuda/Resize.h>
#include <ATen/ops/as_strided.h>
#include <ATen/ops/tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorOptions.h>
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include "paddle/phi/backends/gpu/gpu_info.h"
#endif
#include "ATen/ATen.h"
#include "gtest/gtest.h"
#include "paddle/phi/common/float16.h"
#include "torch/all.h"

namespace {

void DeleteCharArray(void* p) { delete[] static_cast<char*>(p); }

void DeleteIntPtr(void* p) { delete static_cast<int*>(p); }

class RawCompatibleAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t n) override {
    size_t bytes = n == 0 ? 1 : n;
    char* p = new char[bytes];
    return c10::DataPtr(
        p, p, &DeleteCharArray, c10::Device(c10::DeviceType::CPU));
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  c10::DeleterFnPtr raw_deleter() const override { return &DeleteCharArray; }
};

class RawIncompatibleAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t /*n*/) override {
    int* ctx = new int(7);
    void* data = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(ctx) + 1);
    return c10::DataPtr(
        data, ctx, &DeleteIntPtr, c10::Device(c10::DeviceType::CPU));
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  c10::DeleterFnPtr raw_deleter() const override { return &DeleteIntPtr; }
};

class NullRawDeleterAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t n) override {
    size_t bytes = n == 0 ? 1 : n;
    char* p = new char[bytes];
    return c10::DataPtr(
        p, p, &DeleteCharArray, c10::Device(c10::DeviceType::CPU));
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }

  c10::DeleterFnPtr raw_deleter() const override { return nullptr; }
};

class DefaultRawDeleterAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t n) override {
    size_t bytes = n == 0 ? 1 : n;
    char* p = new char[bytes];
    return c10::DataPtr(
        p, p, &DeleteCharArray, c10::Device(c10::DeviceType::CPU));
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    default_copy_data(dest, src, count);
  }
};

}  // namespace

// Regression test (RT-2): tensor.storage() must count the tensor's own
// StorageImpl ownership in use_count(), matching PyTorch where TensorImpl
// holds a Storage handle that participates in use_count.
TEST(StorageTest, StorageUseCountIncludesTensorRef) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage storage = tensor.storage();

  // tensor.storage_ contributes 1, `storage` contributes 1.
  ASSERT_EQ(storage.use_count(), 2)
      << "use_count() must include the tensor's own StorageImpl reference";
  ASSERT_FALSE(storage.unique())
      << "unique() must be false because tensor also holds a reference";
}

// Regression test (RT-3): additional TensorBase wrappers that reference the
// same underlying TensorImpl must not increase Storage owner count.
TEST(StorageTest, StorageUseCountInvariantAcrossIndependentWrappers) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  size_t baseline_count = 0;
  {
    c10::Storage baseline = tensor.storage();
    baseline_count = baseline.use_count();
  }

  // Build a wrapper from the same Paddle tensor impl through a fresh
  // TensorBase constructor path (not TensorBase copy ctor).
  at::TensorBase independent_wrapper(tensor._PD_GetInner());
  c10::Storage current = tensor.storage();

  ASSERT_EQ(current.use_count(), baseline_count)
      << "creating an independent wrapper around the same TensorImpl must not "
         "change storage use_count()";
}

// Regression test (RT-4): storage pointer mutation through one wrapper must
// persist for other wrappers that share the same TensorImpl, even after the
// mutating wrapper and temporary handles are destroyed.
TEST(StorageTest, StorageMutationPersistsAcrossWrappersAfterDestruction) {
  at::TensorBase tensor = at::ones({4}, at::kFloat);
  at::TensorBase peer_wrapper(tensor._PD_GetInner());

  void* new_ptr = nullptr;
  {
    at::TensorBase mutating_wrapper(tensor._PD_GetInner());
    c10::Storage storage = mutating_wrapper.storage();

    RawCompatibleAllocator allocator;
    c10::DataPtr new_data = allocator.allocate(storage.nbytes());
    new_ptr = new_data.get();
    storage.set_data_ptr_noswap(std::move(new_data));
  }

  ASSERT_EQ(peer_wrapper.data_ptr(), new_ptr)
      << "peer wrapper must observe updated storage pointer after mutating "
         "wrapper is destroyed";

  c10::Storage peer_storage = peer_wrapper.storage();
  ASSERT_EQ(peer_storage.data(), new_ptr)
      << "storage() from peer wrapper must keep the updated pointer";
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

TEST(StorageTest, DeviceAndAliasFallbackBranches) {
  c10::Storage empty;
  EXPECT_EQ(empty.device_type(), phi::AllocationType::CPU);
  EXPECT_EQ(empty.device().GetType(), phi::AllocationType::UNDEFINED);
  EXPECT_EQ(empty.use_count(), static_cast<size_t>(0));

  auto base = at::ones({2, 2}, at::kFloat);
  c10::Storage s0 = base.storage();
  c10::Storage s1 = base.storage();
  EXPECT_TRUE(s0.is_alias_of(s1));

  auto holder = s0.ensureTensorHolder();
  (void)holder;
  EXPECT_GE(s0.use_count(), static_cast<size_t>(1));
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
  int* ctx = new int(21);
  c10::DataPtr ptr(ctx, ctx, &DeleteIntPtr, c10::Device(c10::DeviceType::CPU));
  c10::Storage storage1(
      c10::Storage::use_byte_size_t{}, sizeof(int), std::move(ptr), nullptr);
  c10::Storage storage2 = storage1;
  c10::Storage storage3;

  ASSERT_TRUE(c10::isSharedStorageAlias(storage1, storage2));
  ASSERT_TRUE(c10::isSharedStorageAlias(storage2, storage1));

  ASSERT_FALSE(c10::isSharedStorageAlias(storage1, storage3));

  c10::Storage empty_storage;
  ASSERT_FALSE(c10::isSharedStorageAlias(storage1, empty_storage));
  ASSERT_FALSE(c10::isSharedStorageAlias(empty_storage, storage1));
  ASSERT_FALSE(c10::isSharedStorageAlias(empty_storage, empty_storage));
}

TEST(StorageTest, StorageIsAliasOfMethod) {
  // Test Storage::is_alias_of() method
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = tensor1.view({3, 2});
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
  Traits::assignBorrow(another_borrow, borrowed);
  ASSERT_EQ(another_borrow.allocation(), original.allocation());

  // Test destroyBorrow
  Traits::destroyBorrow(borrowed);
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
  c10::Storage taken = Traits::take(to_take);
  ASSERT_EQ(taken.allocation(), alloc);

  // Test getImpl (mutable)
  Traits::pointer_type ptr = Traits::getImpl(taken);
  ASSERT_NE(ptr, nullptr);
  ASSERT_EQ(ptr->allocation(), alloc);

  // Test getImpl (const)
  const c10::Storage& const_taken = taken;
  Traits::const_pointer_type const_ptr = Traits::getImpl(const_taken);
  ASSERT_NE(const_ptr, nullptr);
  ASSERT_EQ(const_ptr->allocation(), alloc);
}

// Custom deleter for testing external DataPtr
static bool g_test_deleter_called = false;
static void* g_test_deleter_context = nullptr;

static void TestDeleter(void* ctx) {
  g_test_deleter_called = true;
  g_test_deleter_context = ctx;
  // In real usage, would free the memory here
}

TEST(StorageTest, ExternalDataPtrUseCount) {
  // Test use_count() semantics for external DataPtr (with deleter)
  // This verifies AC-1: single Storage use_count == 1, copy == 2
  g_test_deleter_called = false;
  g_test_deleter_context = nullptr;

  void* test_ptr = reinterpret_cast<void*>(0x12345678);
  void* test_ctx = reinterpret_cast<void*>(0xABCDEF00);

  // Create external DataPtr with custom deleter
  c10::DataPtr external_ptr(
      test_ptr, test_ctx, &TestDeleter, c10::Device(c10::DeviceType::CPU));

  // Create Storage from external DataPtr
  c10::Storage storage(c10::Storage::use_byte_size_t{},
                       1024,
                       std::move(external_ptr),
                       nullptr,
                       false);

  // Verify single Storage has use_count == 1 and unique() == true
  ASSERT_EQ(storage.use_count(), 1)
      << "Single external DataPtr Storage should have use_count == 1";
  ASSERT_TRUE(storage.unique())
      << "Single external DataPtr Storage should be unique";

  // Copy the storage
  c10::Storage storage_copy(storage);

  // Verify both Storages have use_count == 2
  ASSERT_EQ(storage.use_count(), 2)
      << "Original Storage should have use_count == 2 after copy";
  ASSERT_EQ(storage_copy.use_count(), 2)
      << "Copied Storage should have use_count == 2";
  ASSERT_FALSE(storage.unique())
      << "Original Storage should not be unique after copy";
  ASSERT_FALSE(storage_copy.unique()) << "Copied Storage should not be unique";

  // Verify both point to the same data
  ASSERT_EQ(storage.data(), storage_copy.data());
}

TEST(StorageTest, ExternalDataPtrDeleterPreserved) {
  // Test that data_ptr().get_deleter() returns original deleter (not wrapper)
  // This verifies AC-2: data_ptr() returns original DataPtr with correct
  // deleter
  g_test_deleter_called = false;
  g_test_deleter_context = nullptr;

  void* test_ptr = reinterpret_cast<void*>(0x12345678);
  void* test_ctx = reinterpret_cast<void*>(0xABCDEF00);

  // Create external DataPtr with custom deleter
  c10::DataPtr external_ptr(
      test_ptr, test_ctx, &TestDeleter, c10::Device(c10::DeviceType::CPU));

  // Verify the original DataPtr has our deleter
  ASSERT_EQ(external_ptr.get_deleter(), &TestDeleter);

  // Create Storage from external DataPtr
  c10::Storage storage(c10::Storage::use_byte_size_t{},
                       1024,
                       std::move(external_ptr),
                       nullptr,
                       false);

  // Get the DataPtr from storage
  const c10::DataPtr& data_ptr = storage.data_ptr();

  // Verify get_deleter() returns the original deleter (not a wrapper)
  ASSERT_EQ(data_ptr.get_deleter(), &TestDeleter)
      << "data_ptr().get_deleter() should return original deleter, not wrapper";

  // Verify get_context() returns original context
  ASSERT_EQ(data_ptr.get_context(), test_ctx)
      << "data_ptr().get_context() should return original context";
}

TEST(StorageTest, ExternalDataPtrCopyPreservesDeleter) {
  // Test that copying Storage preserves the original deleter
  g_test_deleter_called = false;
  g_test_deleter_context = nullptr;

  void* test_ptr = reinterpret_cast<void*>(0x12345678);
  void* test_ctx = reinterpret_cast<void*>(0xABCDEF00);

  // Create external DataPtr with custom deleter
  c10::DataPtr external_ptr(
      test_ptr, test_ctx, &TestDeleter, c10::Device(c10::DeviceType::CPU));

  // Create Storage from external DataPtr
  c10::Storage storage(c10::Storage::use_byte_size_t{},
                       1024,
                       std::move(external_ptr),
                       nullptr,
                       false);

  // Copy the storage
  c10::Storage storage_copy(storage);

  // Verify both have the same deleter
  ASSERT_EQ(storage.data_ptr().get_deleter(), &TestDeleter);
  ASSERT_EQ(storage_copy.data_ptr().get_deleter(), &TestDeleter);

  // Verify both have the same context
  ASSERT_EQ(storage.data_ptr().get_context(), test_ctx);
  ASSERT_EQ(storage_copy.data_ptr().get_context(), test_ctx);
}

TEST(StorageTest, ExternalDataPtrMutableDataPtrCoW) {
  // Test CoW behavior for external DataPtr with deleter
  // With single-path design, CoW is skipped for DataPtr with deleter
  g_test_deleter_called = false;

  void* test_ptr = reinterpret_cast<void*>(0x12345678);
  void* test_ctx = reinterpret_cast<void*>(0xABCDEF00);

  // Create external DataPtr with custom deleter
  c10::DataPtr external_ptr(
      test_ptr, test_ctx, &TestDeleter, c10::Device(c10::DeviceType::CPU));

  // Create Storage from external DataPtr
  c10::Storage storage(c10::Storage::use_byte_size_t{},
                       1024,
                       std::move(external_ptr),
                       nullptr,
                       false);

  // Copy the storage (now use_count should be 2)
  c10::Storage storage_copy(storage);
  ASSERT_EQ(storage.use_count(), 2);
  ASSERT_EQ(storage_copy.use_count(), 2);

  // Call mutable_data_ptr() - for external DataPtr with deleter, CoW is skipped
  // because we cannot clone arbitrary deleters
  c10::DataPtr& mutable_dp = storage_copy.mutable_data_ptr();

  // The mutable_data_ptr should still point to the same data
  ASSERT_EQ(mutable_dp.get(), test_ptr);

  // The deleter should still be the original
  ASSERT_EQ(mutable_dp.get_deleter(), &TestDeleter);
}

TEST(StorageTest, DefaultConstructedStorageUseCount) {
  // Test that default constructed storage has use_count == 0
  c10::Storage storage;

  ASSERT_EQ(storage.use_count(), 0)
      << "Default constructed Storage should have use_count == 0";
  ASSERT_FALSE(storage.unique());
  ASSERT_FALSE(storage.valid());
}

TEST(StorageTest, MovedFromStorageIsGracefullyEmpty) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage original = tensor.storage();

  c10::Storage moved(std::move(original));
  ASSERT_TRUE(moved.valid());

  ASSERT_FALSE(original.valid());
  ASSERT_EQ(original.nbytes(), 0UL);
  ASSERT_FALSE(original.resizable());
  ASSERT_EQ(original.data(), nullptr);
  ASSERT_EQ(original.mutable_data(), nullptr);
  ASSERT_EQ(original.allocation(), nullptr);
  ASSERT_EQ(original.allocator(), nullptr);
  ASSERT_EQ(original.device_type(), phi::AllocationType::CPU);
  ASSERT_EQ(original.use_count(), 0UL);
}

TEST(StorageTest, DataPtrCompareExchangeDeleterAndCastContext) {
  int* ctx = new int(42);
  c10::DataPtr dp(ctx, ctx, &DeleteIntPtr, c10::Device(c10::DeviceType::CPU));

  ASSERT_EQ(dp.cast_context<int>(&DeleteIntPtr), ctx);
  ASSERT_EQ(dp.cast_context<int>(&DeleteCharArray), nullptr);

  ASSERT_TRUE(dp.compare_exchange_deleter(&DeleteIntPtr, &DeleteCharArray));
  ASSERT_EQ(dp.get_deleter(), &DeleteCharArray);

  ASSERT_FALSE(dp.compare_exchange_deleter(&DeleteIntPtr, &DeleteIntPtr));
}

TEST(StorageTest, DataPtrUnsafeResetDataAndCtx) {
  c10::DataPtr empty;
  ASSERT_TRUE(empty == nullptr);
  ASSERT_FALSE(empty != nullptr);

  void* p = reinterpret_cast<void*>(0x1234);
  ASSERT_TRUE(empty.unsafe_reset_data_and_ctx(p));
  ASSERT_EQ(empty.get(), p);
  ASSERT_EQ(empty.get_context(), p);

  int* guarded_ctx = new int(5);
  c10::DataPtr guarded(guarded_ctx,
                       guarded_ctx,
                       &DeleteIntPtr,
                       c10::Device(c10::DeviceType::CPU));
  ASSERT_FALSE(
      guarded.unsafe_reset_data_and_ctx(reinterpret_cast<void*>(0x5678)));
}

TEST(StorageTest, DataPtrMoveAndReleaseContextHelpers) {
  int* ctx = new int(9);
  c10::DataPtr dp(ctx, ctx, &DeleteIntPtr, c10::Device(c10::DeviceType::CPU));

  std::unique_ptr<void, c10::DeleterFnPtr> moved_ctx = dp.move_context();
  ASSERT_EQ(moved_ctx.get(), ctx);
  ASSERT_EQ(dp.get_context(), nullptr);

  int* ctx2 = new int(11);
  c10::DataPtr dp2(
      ctx2, ctx2, &DeleteIntPtr, c10::Device(c10::DeviceType::CPU));
  void* released = dp2.release_context();
  ASSERT_EQ(released, ctx2);
  ASSERT_EQ(dp2.get_context(), nullptr);
  DeleteIntPtr(released);
}

TEST(StorageTest, AllocatorRawAllocateAndDeallocate) {
  RawCompatibleAllocator alloc;
  void* raw = alloc.raw_allocate(8);
  ASSERT_NE(raw, nullptr);
  alloc.raw_deallocate(raw);
}

TEST(StorageTest, AllocatorRawAllocateRejectsMismatchedContext) {
  RawIncompatibleAllocator alloc;
  EXPECT_THROW((void)alloc.raw_allocate(8), std::exception);
}

TEST(StorageTest, AllocatorRawDeallocateRequiresDeleter) {
  NullRawDeleterAllocator alloc;
  EXPECT_THROW(alloc.raw_deallocate(reinterpret_cast<void*>(0x1)),
               std::exception);
}

TEST(StorageTest, AllocatorDefaultRawDeleterIsNull) {
  DefaultRawDeleterAllocator alloc;
  ASSERT_EQ(alloc.raw_deleter(), nullptr);
}

TEST(StorageTest, AllocatorCloneCopiesBytes) {
  RawCompatibleAllocator alloc;

  c10::DataPtr src = alloc.allocate(4);
  auto* src_bytes = static_cast<unsigned char*>(src.get());
  src_bytes[0] = 1;
  src_bytes[1] = 2;
  src_bytes[2] = 3;
  src_bytes[3] = 4;

  c10::DataPtr cloned = alloc.clone(src.get(), 4);
  auto* dst_bytes = static_cast<unsigned char*>(cloned.get());

  ASSERT_EQ(dst_bytes[0], 1);
  ASSERT_EQ(dst_bytes[1], 2);
  ASSERT_EQ(dst_bytes[2], 3);
  ASSERT_EQ(dst_bytes[3], 4);
}

TEST(StorageTest, DataPtrHelpersAndAllocatorSimpleDataPtrChecks) {
  // Cover DataPtr(data, Device) ctor path where context is nullptr.
  int value = 7;
  c10::DataPtr dp(&value, c10::Device(c10::DeviceType::CPU));
  ASSERT_EQ(dp.operator->(), &value);
  ASSERT_EQ(dp.get_context(), nullptr);

  // unsafe_set_device is used by callers that update metadata only.
  dp.unsafe_set_device(c10::Device(c10::DeviceType::CPU));
  ASSERT_EQ(dp.device().type(), c10::DeviceType::CPU);

  // PyTorch only treats context==data as a simple DataPtr.
  RawCompatibleAllocator compatible_alloc;
  ASSERT_FALSE(compatible_alloc.is_simple_data_ptr(dp));

  // is_simple_data_ptr: context==data branch.
  c10::DataPtr simple = compatible_alloc.allocate(4);
  ASSERT_TRUE(compatible_alloc.is_simple_data_ptr(simple));

  // is_simple_data_ptr: context!=data branch.
  c10::DataPtr non_simple = RawIncompatibleAllocator().allocate(4);
  ASSERT_FALSE(compatible_alloc.is_simple_data_ptr(non_simple));

  dp.clear();
  ASSERT_EQ(dp.get(), nullptr);
}

TEST(StorageTest, CreateTensorStorageNullAndHolderReuse) {
  c10::Storage empty = c10::Storage::createTensorStorage(nullptr);
  ASSERT_FALSE(empty.valid());

  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  c10::Storage base_storage = tensor.storage();

  // ensureTensorHolder should be stable and reusable.
  auto holder0 = base_storage.ensureTensorHolder();
  auto holder1 = base_storage.ensureTensorHolder();
  ASSERT_NE(holder0, nullptr);
  ASSERT_EQ(holder0, holder1);

  // createTensorStorage should reuse impl when holder is StorageHolderView.
  c10::Storage from_holder = c10::Storage::createTensorStorage(holder0);
  ASSERT_EQ(from_holder.get_impl(), base_storage.get_impl());
}

TEST(StorageTest, MaybeOwnedAndExclusiveTraitsHelpers) {
  c10::Storage src = at::ones({2, 2}, at::kFloat).storage();

  c10::Storage borrowed =
      c10::MaybeOwnedTraits<c10::Storage>::createBorrow(src);
  ASSERT_EQ(borrowed.get_impl(), src.get_impl());

  c10::Storage assigned;
  c10::MaybeOwnedTraits<c10::Storage>::assignBorrow(assigned, borrowed);
  ASSERT_EQ(assigned.get_impl(), src.get_impl());

  ASSERT_EQ(c10::MaybeOwnedTraits<c10::Storage>::referenceFromBorrow(assigned)
                .get_impl(),
            src.get_impl());
  ASSERT_EQ(c10::MaybeOwnedTraits<c10::Storage>::pointerFromBorrow(assigned)
                ->get_impl(),
            src.get_impl());
  ASSERT_TRUE(
      c10::MaybeOwnedTraits<c10::Storage>::debugBorrowIsValid(assigned));
  c10::MaybeOwnedTraits<c10::Storage>::destroyBorrow(assigned);
  ASSERT_FALSE(assigned.valid());

  using ET = c10::ExclusivelyOwnedTraits<c10::Storage>;
  c10::Storage null_repr = ET::nullRepr();
  ASSERT_FALSE(null_repr.valid());

  c10::Storage in_place = ET::createInPlace();
  ASSERT_FALSE(in_place.valid());

  c10::Storage moved = ET::moveToRepr(std::move(src));
  ASSERT_TRUE(moved.valid());

  c10::Storage taken = ET::take(moved);
  ASSERT_TRUE(taken.valid());
  ASSERT_FALSE(moved.valid());

  ASSERT_NE(ET::getImpl(taken), nullptr);
  const c10::Storage& c_taken = taken;
  ASSERT_NE(ET::getImpl(c_taken), nullptr);
}

TEST(StorageTest, SetDataPtrReturnsOldValues) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase other = at::ones({2, 3}, at::kFloat);
  c10::Storage storage = tensor.storage();

  const void* old_ptr = storage.data();
  auto old_alloc = storage.allocation();
  auto new_alloc = other.storage().allocation();
  ASSERT_NE(new_alloc, nullptr);

  // shared_ptr overload returns previous allocation.
  auto returned_alloc = storage.set_data_ptr(new_alloc);
  ASSERT_EQ(returned_alloc, old_alloc);
  ASSERT_EQ(storage.allocation(), new_alloc);

  // DataPtr overload returns previous DataPtr and clears allocation-backed
  // state on write.
  c10::DataPtr new_data_ptr(other.data_ptr(),
                            c10::Device(c10::DeviceType::CPU));
  c10::DataPtr old_data_ptr = storage.set_data_ptr(std::move(new_data_ptr));
  ASSERT_EQ(old_data_ptr.get(), new_alloc->ptr());
  ASSERT_EQ(storage.data(), other.data_ptr());
  ASSERT_EQ(storage.allocation(), nullptr);
  ASSERT_NE(old_data_ptr.get(), old_ptr);
}

// ---------------------------------------------------------------------------
// Reference Semantics Tests
//
// These tests verify that Storage copies share a single underlying
// StorageImpl, so mutations via set_data_ptr*(), set_nbytes(), and
// mutable_data_ptr() are visible through all handles — matching the
// observable contract of PyTorch's c10::Storage (which wraps a shared
// StorageImpl via intrusive_ptr).
// ---------------------------------------------------------------------------

TEST(StorageTest, ReferenceSemanticsMutationVisibleThroughCopy) {
  // After copying a Storage, writing to one handle is visible via the other.
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = at::ones({4, 5}, at::kFloat);

  c10::Storage storage_a = tensor1.storage();
  c10::Storage storage_b = storage_a;  // shares StorageImpl

  ASSERT_EQ(storage_a.data(), storage_b.data())
      << "Copies should start with the same data pointer";

  // Replace allocation via the shared_ptr overload
  auto new_alloc = tensor2.storage().allocation();
  storage_a.set_data_ptr_noswap(new_alloc);

  ASSERT_EQ(storage_b.allocation(), new_alloc)
      << "storage_b should see the allocation change made through storage_a";
  ASSERT_EQ(storage_a.data(), storage_b.data())
      << "Both handles should point to the same data after mutation";
}

TEST(StorageTest, ReferenceSemanticsMutableDataPtrShared) {
  // mutable_data_ptr() returns a reference into the shared StorageImpl,
  // so the reference obtained from one handle is the same object as the
  // data_ptr() accessed through its copy.
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  c10::Storage storage_a = tensor.storage();
  c10::Storage storage_b = storage_a;

  c10::DataPtr& dp_via_a = storage_a.mutable_data_ptr();

  ASSERT_EQ(&dp_via_a, &storage_b.data_ptr())
      << "mutable_data_ptr() from one handle should be the same object as "
         "data_ptr() from another handle that shares the StorageImpl";
}

TEST(StorageTest, ReferenceSemanticsMutationNotVisibleAcrossIndependent) {
  // Two Storage objects constructed independently (not by copying one from the
  // other) do NOT share a StorageImpl, so mutations through one do not affect
  // the other.
  at::TensorBase tensor1 = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = at::ones({4, 5}, at::kFloat);

  // Two independently-created Storages — different StorageImpls
  c10::Storage storage_a = tensor1.storage();
  c10::Storage storage_b = tensor2.storage();

  const void* original_b_data = storage_b.data();
  auto new_alloc = tensor2.storage().allocation();
  storage_a.set_data_ptr_noswap(new_alloc);

  ASSERT_EQ(storage_b.data(), original_b_data)
      << "Independently-constructed Storage should not be affected by "
         "mutations to another Storage";
}

TEST(StorageTest, ReferenceSemanticsTwoIndependentStorageCalls) {
  // Multiple calls to tensor.storage() on the same tensor return handles
  // sharing the same underlying StorageImpl, matching PyTorch's reference
  // semantics where TensorBase::storage() always refers to the same storage.
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase tensor2 = at::ones({4, 5}, at::kFloat);

  c10::Storage storage_b = tensor.storage();
  c10::Storage storage_c = tensor.storage();  // same impl as storage_b

  // Both handles point to the same underlying data.
  ASSERT_EQ(storage_b.data(), storage_c.data())
      << "Two Storage handles from the same tensor should share the same data "
         "pointer initially";

  // Mutation through one handle is visible through the other.
  auto new_alloc = tensor2.storage().allocation();
  storage_b.set_data_ptr_noswap(new_alloc);

  ASSERT_EQ(storage_c.data(), storage_b.data())
      << "Mutation through one Storage handle should be visible through "
         "another handle obtained from the same tensor";
}

TEST(StorageTest, StorageMutationUpdatesTensorDataPtr) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase other = at::ones({4, 5}, at::kFloat);

  c10::Storage storage = tensor.storage();
  auto new_alloc = other.storage().allocation();
  ASSERT_NE(new_alloc, nullptr);
  ASSERT_NE(tensor.data_ptr(), new_alloc->ptr());

  storage.set_data_ptr_noswap(new_alloc);

  ASSERT_EQ(tensor.data_ptr(), new_alloc->ptr())
      << "tensor.data_ptr() must follow mutations made through "
         "tensor.storage()";
  ASSERT_EQ(tensor.storage().allocation(), new_alloc)
      << "Repeated storage() calls should observe the live allocation";
}

TEST(StorageTest, StorageMutationPersistsAfterHandleDestruction) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase other = at::ones({4, 5}, at::kFloat);

  auto new_alloc = other.storage().allocation();
  ASSERT_NE(new_alloc, nullptr);
  {
    c10::Storage storage = tensor.storage();
    storage.set_data_ptr_noswap(new_alloc);
  }

  ASSERT_EQ(tensor.data_ptr(), new_alloc->ptr())
      << "Tensor should keep the storage alive after external handles die";
  ASSERT_EQ(tensor.storage().allocation(), new_alloc);
}

TEST(StorageTest, RepeatedStorageCallsReturnSameReference) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  const c10::Storage& storage_a = tensor.storage();
  const c10::Storage& storage_b = tensor.storage();

  ASSERT_EQ(&storage_a, &storage_b)
      << "storage() must return the same reference for one tensor wrapper";
  ASSERT_EQ(storage_a.get_impl(), storage_b.get_impl());
}

TEST(StorageTest, CopiedTensorWrappersShareStorageImpl) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase alias = tensor;
  at::TensorBase other = at::ones({4, 5}, at::kFloat);

  auto new_alloc = other.storage().allocation();
  ASSERT_NE(new_alloc, nullptr);

  c10::Storage storage = tensor.storage();
  storage.set_data_ptr_noswap(new_alloc);

  ASSERT_EQ(tensor.storage().get_impl(), alias.storage().get_impl());
  ASSERT_EQ(alias.data_ptr(), new_alloc->ptr())
      << "Copied TensorBase wrappers must observe shared storage mutations";
}

TEST(StorageTest, AsStridedViewSharesStorageImplWithBaseTensor) {
  at::Tensor base = at::tensor({1, 2, 3, 4}, at::kInt);
  at::Tensor view = base.as_strided({3}, {1}, 1);

  ASSERT_EQ(base.storage().get_impl(), view.storage().get_impl());

  view.resize_({4});

  ASSERT_EQ(view.data_ptr<int>(), base.data_ptr<int>() + 1)
      << "Growing an as_strided view must update the shared compat storage "
         "visible from the base tensor";
}

TEST(StorageTest, AliasWrapperDoesNotIncreaseTensorOwnedStorageCount) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  const c10::Storage& single_wrapper_storage = tensor.storage();
  size_t single_wrapper_count = single_wrapper_storage.use_count();
  ASSERT_GE(single_wrapper_count, 1UL);

  at::TensorBase alias = tensor;
  const c10::Storage& alias_storage = alias.storage();

  ASSERT_EQ(single_wrapper_storage.get_impl(), alias_storage.get_impl());
  ASSERT_EQ(single_wrapper_storage.use_count(), single_wrapper_count)
      << "A copied TensorBase wrapper sharing the same tensor impl must not "
         "add an extra tensor-owned Storage reference";
}

TEST(StorageTest, ViewTensorWrappersShareStorageImpl) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  at::TensorBase alias = tensor.view({3, 2});

  c10::Storage tensor_storage = tensor.storage();
  c10::Storage alias_storage = alias.storage();

  ASSERT_TRUE(tensor_storage.is_alias_of(alias_storage))
      << "Fresh wrappers over the same underlying storage should share a "
         "StorageImpl";
  ASSERT_FALSE(c10::isSharedStorageAlias(tensor_storage, alias_storage))
      << "isSharedStorageAlias() should follow DataPtr ownership semantics, "
         "not overlapping ranges";
}

TEST(StorageTest, HasStorageTracksLiveStorageState) {
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);
  ASSERT_TRUE(tensor.has_storage());

  c10::Storage storage = tensor.storage();
  storage.set_data_ptr(at::DataPtr());
  ASSERT_FALSE(tensor.has_storage());

  tensor.reset();
  ASSERT_FALSE(tensor.has_storage());
}

TEST(StorageTest, ReferenceSemanticsSetNbytesVisibleThroughCopy) {
  // set_nbytes() on one handle is visible through its copy.
  at::TensorBase tensor = at::ones({2, 3}, at::kFloat);

  c10::Storage storage_a = tensor.storage();
  c10::Storage storage_b = storage_a;

  size_t new_size = 42;
  storage_a.set_nbytes(new_size);

  ASSERT_EQ(storage_b.nbytes(), new_size)
      << "set_nbytes() change should be visible through all copies";
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(StorageTest, CUDAAllocatorZeroBytePreservesDevice) {
  // getCUDADeviceAllocator()->allocate(0) must return a DataPtr whose device
  // is the current CUDA device, not a default-constructed CPU DataPtr.
  if (!at::cuda::is_available()) {
    return;  // No CUDA device, skip
  }

  c10::Allocator* alloc = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(alloc, nullptr);

  c10::DataPtr dp = alloc->allocate(0);

  // Pointer should be null for zero-byte allocation
  ASSERT_EQ(dp.get(), nullptr)
      << "Zero-byte allocation should return null pointer";

  // Device type must be CUDA, not CPU.  For HIP/ROCm builds, PyTorch's
  // compatibility convention is to expose DeviceType::CUDA rather than a
  // separate HIP device type, so we follow the same convention.
  ASSERT_EQ(dp.device().type(), c10::DeviceType::CUDA)
      << "Zero-byte CUDA allocator DataPtr should carry CUDA device type";

  // Device index should match the current device
  int current_device = phi::backends::gpu::GetCurrentDeviceId();
  ASSERT_EQ(static_cast<int>(dp.device().index()), current_device)
      << "Zero-byte DataPtr should carry the current CUDA device index";
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(StorageTest, CUDAAllocatorRawDeleterIsNull) {
  // PaddleCUDAAllocatorAdapter::raw_deleter() must return nullptr because the
  // c10::Allocator raw API contract requires get()==get_context() in the
  // returned DataPtr, but our adapter returns data=device_ptr,
  // context=phi::Allocation*, which violates that contract.
  // Returning nullptr signals that raw_allocate/raw_deallocate are unsafe.
  c10::Allocator* alloc = at::cuda::getCUDADeviceAllocator();
  ASSERT_NE(alloc, nullptr);
  ASSERT_EQ(alloc->raw_deleter(), nullptr)
      << "PaddleCUDAAllocatorAdapter::raw_deleter() must return nullptr "
         "because get() != get_context() in its allocate() DataPtr";
}
#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP
