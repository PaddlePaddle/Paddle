// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/allocation/pile_allocator.h"
#include <gtest/gtest.h>
#include <time.h>
#include <unordered_set>

namespace paddle {
namespace memory {
namespace allocation {

TEST(BitSet, main) {
  BitSet bits(11);
  ASSERT_EQ(bits.num_bytes(), 2);

  for (int i = 0; i < 11; i++) {
    ASSERT_FALSE(bits.Tell(i));
  }

  // Set one
  bits.Set(3);
  ASSERT_TRUE(bits.Tell(3));

  bits.Flip(3);
  ASSERT_FALSE(bits.Tell(3));
  bits.Flip(3);
  ASSERT_TRUE(bits.Tell(3));

  bits.Flip(15);
  EXPECT_TRUE(bits.Tell(15));
  bits.Flip(15);
  EXPECT_FALSE(bits.Tell(15));
}

TEST(StackSet, test) {
  StackSet<int> stack;
  for (int i = 0; i < 10; i++) {
    stack.push(i);
  }

  ASSERT_TRUE(stack.count(5));
  ASSERT_TRUE(stack.count(8));

  stack.pop(4);
  ASSERT_FALSE(stack.count(4));
}

TEST(BuddyResource, test) {
  auto resource = CreateBuddyResource(10, 2, 9, 2);
  ASSERT_EQ(resource->num_buckets(), 10 - 2 + 1);
}

TEST(BuddySystem, test1) {
  BuddySystem manager(4, 10, 9);
  LOG(INFO) << "min mem: " << manager.min_mem_size_ << " "
            << manager.min_mem_log_size_;
  LOG(INFO) << "max mem: " << manager.max_mem_size_ << " "
            << manager.max_mem_log_size_;
}

TEST(BuddySystem, BucketForRequest) {
  BuddySystem manager(4, 10, 9);
  LOG(INFO) << "num buckets: " << manager.num_buckets_;
  ASSERT_EQ(manager.BucketForRequest(1 << 10), 0);
  ASSERT_EQ(manager.BucketForRequest(1 << 9), 1);
  for (int i = 0; i < 1 << 8; i++) {
    ASSERT_EQ(manager.BucketForRequest((1 << 9) - i), 1);
  }
}

TEST(BuddySystem, BucketSize) {
  BuddySystem manager(4, 10, 9);
  ASSERT_EQ(manager.BucketSize(0), 1 << 10);
  ASSERT_EQ(manager.BucketSize(1), 1 << 9);
}

TEST(BuddySystem, TestFirstBucket) {
  BuddySystem manager(4, 10, 9);
  ASSERT_EQ(manager.buckets_->size(), manager.num_buckets_);
  // Only the first one bucket has one element.
  for (size_t i = 1; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(!manager.buckets(i).empty());
  }
  ASSERT_TRUE(!manager.buckets_->front().empty());
  ASSERT_EQ(manager.buckets_->front().size(), 1UL);
  // Pop the only one bucket
  auto* res = manager.PopBucket(0);
  ASSERT_TRUE(res);
  // The first bucket just align from the head of the buffer.
  ASSERT_EQ(reinterpret_cast<uint8_t*>(res), manager.buffer(0));
  // Pop if there is no remaining buckets.
  ASSERT_FALSE(manager.PopBucket(0));
}

TEST(BuddySystem, PushBucket) {
  BuddySystem manager(4, 10, 9);

  // Push to the first bucket.
  ASSERT_FALSE(!manager.buckets(1).empty());

  manager.PushBucket(1, manager.buffer_->front());
  ASSERT_EQ(manager.buckets(1).size(), 1UL);

  // There are 1<<10 nodes for the 0th bucket.
  for (int i = 2; i < 7; i++) {
    manager.PushBucket(1, manager.buffer_->front());
    ASSERT_EQ(manager.buckets(1).size(), i);
  }
}

TEST(BuddySystem, PopBucket) {
  BuddySystem manager(4, 10, 9);
  for (int i = 0; i < 4; i++) {
    manager.PushBucket(0, manager.buffer_->front());
  }
  ASSERT_EQ(manager.buckets(0).size(), 4 + 1);

  for (int i = 0; i < 4; i++) {
    manager.PopBucket(0);
    ASSERT_EQ(manager.buckets(0).size(), 4 - i);
  }
}

TEST(BuddySystem, GetBrotherNode) {
  BuddySystem manager(4, 10, 9);
  ASSERT_EQ(manager.GetBrotherNode(1), 2);
  ASSERT_EQ(manager.GetBrotherNode(3), 4);
  ASSERT_EQ(manager.GetBrotherNode(4), 3);
  ASSERT_EQ(manager.GetBrotherNode(5), 6);
  ASSERT_EQ(manager.GetBrotherNode(6), 5);
  ASSERT_EQ(manager.GetBrotherNode(7), 8);
  ASSERT_EQ(manager.GetBrotherNode(8), 7);
}

TEST(BuddySystem, SplitBucket) {
  BuddySystem manager(4, 10, 9);
  ASSERT_EQ(manager.buckets(0).size(), 1UL);
  manager.SplitBucket(0, manager.buffer(0), (1 << 10) / 2, 0);
  // The large bucket will not poped by default.
  ASSERT_EQ(manager.buckets(1).size(), 1UL);
  ASSERT_EQ(reinterpret_cast<uint8_t*>(manager.buckets(1).top()),
            manager.buffer(0) + (1 << 10) / 2);

  ASSERT_TRUE(manager.is_splits(0).Tell(0));
}

TEST(BuddySystem, NodeForPtr) {
  BuddySystem manager(4, 10, 9);
  ASSERT_EQ(manager.NodeForPtr(manager.buffer_->front(),
                               manager.BucketForRequest(1 << 10), 0),
            0UL);

  ASSERT_EQ(manager.NodeForPtr(manager.buffer_->front(),
                               manager.BucketForRequest(1 << 9), 0),
            1UL);
  ASSERT_EQ(manager.NodeForPtr(manager.buffer_->front() + (1 << 9),
                               manager.BucketForRequest(1 << 9), 0),
            2UL);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_->front() + i * (1 << 8),
                                 manager.BucketForRequest(1 << 8), 0),
              2UL + i + 1);
  }

  for (int i = 0; i < 4 * 2; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_->front() + i * (1 << 7),
                                 manager.BucketForRequest(1 << 7), 0),
              6UL + i + 1);
  }
}

TEST(BuddySystem, PtrForNode) {
  BuddySystem manager(4, 10, 9);
  // same start point with different buddy size.
  ASSERT_EQ(manager.PtrForNode(0, 0, 0), manager.buffer(0));
  ASSERT_EQ(manager.PtrForNode(1, 1, 0), manager.buffer(0));
  ASSERT_EQ(manager.PtrForNode(2, 1, 0), manager.buffer(0) + (1 << 9));
  ASSERT_EQ(manager.PtrForNode(3, 2, 0), manager.buffer(0));
  ASSERT_EQ(manager.PtrForNode(4, 2, 0), manager.buffer(0) + 1 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(5, 2, 0), manager.buffer(0) + 2 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(6, 2, 0), manager.buffer(0) + 3 * (1 << 8));
}

TEST(BuddySystem, Malloc) {
  BuddySystem manager(4, 10, 9, false);
  // raw system allocation.
  auto raw_ptr = manager.Malloc(1 << 11);
  ASSERT_EQ(-1, manager.PoolIdxForPtr(static_cast<uint8_t*>(raw_ptr)));

  auto* ptr = manager.Malloc(1 << 9);
  ASSERT_EQ(ptr, manager.buffer(0));
  // The 1-th bucket should not be empty
  ASSERT_TRUE(!manager.buckets(1).empty());
  for (size_t i = 2; i < manager.num_buckets_; i++) {
    ASSERT_TRUE(manager.buckets(i).top());
  }

  ptr = manager.Malloc((1 << 9));
  ASSERT_EQ(ptr, manager.buffer(0) + (1 << 9));

  for (size_t i = 1; i < manager.num_buckets_; i++) {
    ASSERT_TRUE(manager.buckets(i).top());
  }

  // OOM
  ASSERT_FALSE(manager.Malloc(1 << 2));
}

TEST(BuddySystem, Malloc1) {
  BuddySystem manager(2, 10, 9);

  auto ShowBuckets = [&] {
    for (size_t i = 0; i < manager.num_buckets_; i++) {
      LOG(INFO) << i << " "
                << " bucket_size " << manager.BucketSize(i);
    }

    for (int v : std::vector<int>({0, 1, 3, 7, 8, 15})) {
      LOG(INFO) << "node is flip " << v << " " << manager.is_splits(0).Tell(v);
    }
  };

  auto CheckIsSplits = [&](const std::unordered_set<int>& nodes) {
    for (size_t i = 0; i < manager.is_splits(0).num_bytes(); i++) {
      if (nodes.count(i)) {
        if (!manager.is_splits(0).Tell(i)) {
          LOG(ERROR) << "node " << i << " != true";
        }
        EXPECT_TRUE(manager.is_splits(0).Tell(i));
      } else {
        if (manager.is_splits(0).Tell(i)) {
          LOG(ERROR) << "node " << i << " != false";
        }
        EXPECT_FALSE(manager.is_splits(0).Tell(i));
      }
    }
  };

  LOG(INFO) << "allocate " << (1 << 6) << " ----------------------------";
  // need extra 1<<3 if Header included, so an 1<<6 block is needed.
  // The node 15 is used.
  auto* node_15 = manager.Malloc(1 << 6);
  ShowBuckets();
  CheckIsSplits({0, 1, 3, 7});
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 1);  // 1<<9
  ASSERT_EQ(manager.buckets(2).size(), 1);  // 1<<8
  ASSERT_EQ(manager.buckets(3).size(), 1);  // 1<<7
  ASSERT_EQ(manager.buckets(4).size(), 1);  // 1<<6

  LOG(INFO) << "allocate " << (1 << 6) << " ----------------------------";
  // Alloc another 1<<6 block, the node 16 is used, so the whole node 7 is used.
  auto* node_16 = manager.Malloc(1 << 6);
  ShowBuckets();
  CheckIsSplits({0, 1, 3});
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 1);  // 1<<9
  ASSERT_EQ(manager.buckets(2).size(), 1);  // 1<<8
  ASSERT_EQ(manager.buckets(3).size(), 1);  // 1<<7
  ASSERT_EQ(manager.buckets(4).size(), 0);  // 1<<6

  // Alloc another 1<<6 block, the node 17 is used.
  auto* node_17 = manager.Malloc(1 << 6);
  ShowBuckets();
  CheckIsSplits({0, 1, 8});  // node 3 is used, so not split
  // Alloc another 1<<6 block, the node 18 is used, and the whole node 8 is
  // used.
  // auto* node_18 =
  manager.Malloc(1 << 6);
  ShowBuckets();
  CheckIsSplits({0, 1});  // node 3 is used, so not split

  // Free node 17
  manager.Free(node_17);
  ShowBuckets();
  CheckIsSplits({0, 1, 8});
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 1);  // 1<<9
  ASSERT_EQ(manager.buckets(2).size(), 1);  // 1<<8
  ASSERT_EQ(manager.buckets(3).size(), 0);  // 1<<7
  ASSERT_EQ(manager.buckets(4).size(), 1);  // 1<<6, node17

  // Free node 15
  manager.Free(node_15);
  ShowBuckets();
  CheckIsSplits({0, 1, 7, 8});
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 1);  // 1<<9
  ASSERT_EQ(manager.buckets(2).size(), 1);  // 1<<8
  ASSERT_EQ(manager.buckets(3).size(), 0);  // 1<<7
  ASSERT_EQ(manager.buckets(4).size(), 2);  // 1<<6, node17 and node15

  // Free node 16, that block should merge with the block of node15, and make
  // the entire node7 free.
  manager.Free(node_16);
  ShowBuckets();
  CheckIsSplits({0, 1, 3, 8});  // node7 is free, and make node3 split
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 1);  // 1<<9
  ASSERT_EQ(manager.buckets(2).size(), 1);  // 1<<8
  ASSERT_EQ(manager.buckets(3).size(), 1);  // 1<<7
  ASSERT_EQ(manager.buckets(4).size(), 1);  // 1<<6, node17 and node15

  // auto* node_4 =
  manager.Malloc((1 << 8));
  ShowBuckets();
  CheckIsSplits({0, 3, 8});

  // Alloc node5 and node6, and that will make node2 used.
  // auto* node_5 =
  manager.Malloc((1 << 8));
  // auto* node_6 =
  manager.Malloc((1 << 8));
  ShowBuckets();
  CheckIsSplits({3, 8});
  ASSERT_EQ(manager.buckets(0).size(), 0);
  ASSERT_EQ(manager.buckets(1).size(), 0);  // 1<<9, node2 is used.
}

TEST(BuddySystem, realloc) {
  BuddySystem manager(2, 10, 9, true);

  auto ShowBuckets = [&] {
    for (size_t i = 0; i < manager.num_buckets_; i++) {
      LOG(INFO) << i << " " << manager.buckets(i).size() << " bucket_size "
                << manager.BucketSize(i);
    }
  };

  // Occupy the initial memory block.
  auto* ptr = manager.Malloc((1 << 10));
  ASSERT_EQ(manager.PoolIdxForPtr(static_cast<uint8_t*>(ptr)), 0);
  ASSERT_TRUE(ptr);
  ASSERT_EQ(manager.is_splits_->size(), 1UL);
  ASSERT_EQ(manager.buffer_->size(), 1UL);
  ShowBuckets();

  auto* ptr1 = manager.Malloc((1 << 8));
  ASSERT_TRUE(ptr1);
  ASSERT_EQ(manager.is_splits_->size(), 2UL);
  ASSERT_EQ(manager.buffer_->size(), 2UL);
  ASSERT_EQ(manager.PoolIdxForPtr(static_cast<uint8_t*>(ptr1)), 1);

  // auto* ptr2 =
  manager.Malloc((1 << 8));
  ASSERT_EQ(manager.PoolIdxForPtr(static_cast<uint8_t*>(ptr1)), 1);
}

TEST(BuddySystem, resource_created_externally) {
  auto resource = CreateBuddyResource(10, 2, 9, 4);
  ASSERT_EQ(resource->num_piled(), 4);
}

TEST(PileAllocator, crazy_alloc) {
  PileAllocator::MemoryOption option(1 << 10, 1 << 4, 1 << 10);
  PileAllocator allocator(0, option, option);

  void* last_ptr{nullptr};

  for (int i = 0; i < 100; i++) {
    if (i == 0 || i % 3 != 0) {
      size_t size = rand() % 800;
      last_ptr = allocator.RawAllocate(size);
      LOG(INFO) << "allocate " << size << " : " << last_ptr;
    } else {
      allocator.RawFree(last_ptr);
      LOG(INFO) << "free "
                << " : " << last_ptr;
    }
    LOG(INFO) << "frag ratio: " << allocator.frag_ratio(-1);
  }
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
