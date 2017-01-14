/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include "paddle/utils/Logging.h"
#include "paddle/utils/Util.h"
#define private public
#include "paddle/math/Allocator.h"
#include "paddle/math/MemoryHandle.h"
#include "paddle/math/PoolAllocator.h"

using namespace paddle;  // NOLINT

template <typename Allocator>
void testPoolAllocator() {
  PoolAllocator* pool =
      new PoolAllocator(new Allocator(), /* sizeLimit */ 1024);

  /* alloc from system memory */
  void* ptr1 = pool->alloc(10);
  void* ptr2 = pool->alloc(200);
  void* ptr3 = pool->alloc(200);
  pool->free(ptr1, 10);
  pool->free(ptr2, 200);
  pool->free(ptr3, 200);
  pool->printAll();
  EXPECT_EQ((size_t)2, pool->pool_.size());
  EXPECT_EQ((size_t)1, pool->pool_[10].size());
  EXPECT_EQ((size_t)2, pool->pool_[200].size());
  EXPECT_EQ(ptr1, pool->pool_[10][0]);
  EXPECT_EQ(ptr2, pool->pool_[200][0]);
  EXPECT_EQ(ptr3, pool->pool_[200][1]);

  /* alloc from pool */
  void* ptr4 = pool->alloc(10);
  void* ptr5 = pool->alloc(200);
  pool->printAll();
  EXPECT_EQ((size_t)0, pool->pool_[10].size());
  EXPECT_EQ((size_t)1, pool->pool_[200].size());
  EXPECT_EQ(ptr1, ptr4);
  EXPECT_EQ(ptr3, ptr5);
  pool->free(ptr4, 10);
  pool->free(ptr5, 200);

  /* alloc size > sizeLimit */
  void* ptr6 = pool->alloc(1024);
  pool->free(ptr6, 1024);
  EXPECT_LE((size_t)1024, pool->poolMemorySize_);

  void* ptr7 = pool->alloc(1);
  EXPECT_EQ((size_t)0, pool->poolMemorySize_);
  EXPECT_EQ((size_t)0, pool->pool_.size());
  pool->free(ptr7, 1);

  delete pool;
}

TEST(Allocator, Pool) {
  testPoolAllocator<CpuAllocator>();
#ifndef PADDLE_ONLY_CPU
  testPoolAllocator<GpuAllocator>();
#endif
}

TEST(MemoryHandle, Cpu) {
  for (auto size : {10, 30, 50, 100, 200, 512, 1000, 1023, 1024, 1025, 8193}) {
    CpuMemoryHandle handle(size);
    EXPECT_LE(handle.getSize(), handle.getAllocSize());
  }

  void* ptr1;
  void* ptr2;
  {
    CpuMemoryHandle handle(256);
    ptr1 = handle.getBuf();
  }
  {
    CpuMemoryHandle handle(256);
    ptr2 = handle.getBuf();
  }
  EXPECT_EQ(ptr1, ptr2);
}

#ifndef PADDLE_ONLY_CPU
TEST(MemoryHandle, Gpu) {
  int numGpu = hl_get_device_count();

  /* alloc from system memory */
  void* ptr3[numGpu];
  void* ptr4[numGpu];
  for (int i = 0; i < numGpu; i++) {
    SetDevice device(i);
    GpuMemoryHandle handle1(30);
    GpuMemoryHandle handle2(30);
    GpuMemoryHandle handle3(4000);
    GpuMemoryHandle handle4(500);
    ptr3[i] = handle3.getBuf();
    ptr4[i] = handle4.getBuf();
  }

  /* alloc from pool */
  for (int i = 0; i < numGpu; i++) {
    SetDevice device(i);
    GpuMemoryHandle handle1(30);
    GpuMemoryHandle handle3(4000);
    GpuMemoryHandle handle4(500);
    EXPECT_EQ(ptr3[i], handle3.getBuf());
    EXPECT_EQ(ptr4[i], handle4.getBuf());
  }
}
#endif
