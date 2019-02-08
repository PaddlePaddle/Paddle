#include "paddle/fluid/memory/allocation/pile_allocator.h"
#include <gtest/gtest.h>
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

TEST(BuddyManager, test1) {
  BuddyManager manager(4, 10, 9);
  LOG(INFO) << "min mem: " << manager.min_mem_size_ << " "
            << manager.min_mem_log_size_;
  LOG(INFO) << "max mem: " << manager.max_mem_size_ << " "
            << manager.max_mem_log_size_;
}

TEST(BuddyManager, BucketForRequest) {
  BuddyManager manager(4, 10, 9);
  LOG(INFO) << "num buckets: " << manager.num_buckets_;
  ASSERT_EQ(manager.BucketForRequest(1 << 10), 0);
  ASSERT_EQ(manager.BucketForRequest(1 << 9), 1);
  for (int i = 0; i < 1 << 8; i++) {
    ASSERT_EQ(manager.BucketForRequest((1 << 9) - i), 1);
  }
}

TEST(BuddyManager, BucketSize) {
  BuddyManager manager(4, 10, 9);
  ASSERT_EQ(manager.BucketSize(0), 1 << 10);
  ASSERT_EQ(manager.BucketSize(1), 1 << 9);
}

TEST(BuddyManager, TestFirstBucket) {
  BuddyManager manager(4, 10, 9);
  ASSERT_EQ(manager.buckets_.size(), manager.num_buckets_);
  // Only the first one bucket has one element.
  for (int i = 1; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(manager.buckets_[i]);
  }
  ASSERT_TRUE(manager.buckets_.front());
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_.front()), 1UL);
  // Pop the only one bucket
  auto* res = manager.PopBucket(0);
  ASSERT_TRUE(res);
  // The first bucket just align from the head of the buffer.
  ASSERT_EQ(reinterpret_cast<uint8_t*>(res), manager.buffer_[0]);
  ASSERT_EQ(res->next, res);
  ASSERT_EQ(res->pre, res);
  // Pop if there is no remaining buckets.
  ASSERT_FALSE(manager.PopBucket(0));
}

TEST(BuddyManager, PushBucket) {
  BuddyManager manager(4, 10, 9);

  // Push to the first bucket.
  ASSERT_FALSE(manager.buckets_[1]);

  manager.PushBucket(1, reinterpret_cast<ListNode*>(manager.buffer_.front() +
                                                    sizeof(ListNode)));
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1UL);
  ASSERT_EQ(manager.buckets_[1]->next, manager.buckets_[1]);
  ASSERT_EQ(manager.buckets_[1]->pre, manager.buckets_[1]);

  // There are 1<<10 nodes for the 0th bucket.
  for (int i = 2; i < 7; i++) {
    manager.PushBucket(1, reinterpret_cast<ListNode*>(manager.buffer_.front() +
                                                      i * sizeof(ListNode)));
    ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), i);
  }
}

TEST(BuddyManager, PopBucket) {
  BuddyManager manager(4, 10, 9);
  for (int i = 0; i < 4; i++) {
    manager.PushBucket(0, reinterpret_cast<ListNode*>(manager.buffer_.front() +
                                                      i * sizeof(ListNode)));
  }

  for (int i = 0; i < 4; i++) {
    manager.PopBucket(0);
    ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 4 - i - 1);
  }
}

TEST(BuddyManager, GetBrotherNode) {
  BuddyManager manager(4, 10, 9);
  ASSERT_EQ(manager.GetBrotherNode(1), 2);
  ASSERT_EQ(manager.GetBrotherNode(3), 4);
  ASSERT_EQ(manager.GetBrotherNode(4), 3);
  ASSERT_EQ(manager.GetBrotherNode(5), 6);
  ASSERT_EQ(manager.GetBrotherNode(6), 5);
  ASSERT_EQ(manager.GetBrotherNode(7), 8);
  ASSERT_EQ(manager.GetBrotherNode(8), 7);
}

TEST(BuddyManager, SplitBucket) {
  BuddyManager manager(4, 10, 9);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 1UL);
  manager.SplitBucket(0, manager.buffer_[0], (1 << 10) / 2, 0);
  // The large bucket will not poped by default.
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1UL);
  ASSERT_EQ(reinterpret_cast<uint8_t*>(manager.buckets_[1]),
            manager.buffer_[0] + (1 << 10) / 2);

  ASSERT_TRUE(manager.is_splits_[0].Tell(0));
}

TEST(BuddyManager, NodeForPtr) {
  BuddyManager manager(4, 10, 9);
  ASSERT_EQ(manager.NodeForPtr(manager.buffer_.front(),
                               manager.BucketForRequest(1 << 10), 0),
            0UL);

  ASSERT_EQ(manager.NodeForPtr(manager.buffer_.front(),
                               manager.BucketForRequest(1 << 9), 0),
            1UL);
  ASSERT_EQ(manager.NodeForPtr(manager.buffer_.front() + (1 << 9),
                               manager.BucketForRequest(1 << 9), 0),
            2UL);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_.front() + i * (1 << 8),
                                 manager.BucketForRequest(1 << 8), 0),
              2UL + i + 1);
  }

  for (int i = 0; i < 4 * 2; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_.front() + i * (1 << 7),
                                 manager.BucketForRequest(1 << 7), 0),
              6UL + i + 1);
  }
}

TEST(BuddyManager, PtrForNode) {
  BuddyManager manager(4, 10, 9);
  // same start point with different buddy size.
  ASSERT_EQ(manager.PtrForNode(0, 0, 0), manager.buffer_[0]);
  ASSERT_EQ(manager.PtrForNode(1, 1, 0), manager.buffer_[0]);
  ASSERT_EQ(manager.PtrForNode(2, 1, 0), manager.buffer_[0] + (1 << 9));
  ASSERT_EQ(manager.PtrForNode(3, 2, 0), manager.buffer_[0]);
  ASSERT_EQ(manager.PtrForNode(4, 2, 0), manager.buffer_[0] + 1 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(5, 2, 0), manager.buffer_[0] + 2 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(6, 2, 0), manager.buffer_[0] + 3 * (1 << 8));
}

TEST(BuddyManager, Malloc) {
  BuddyManager manager(4, 10, 9, false);
  // raw system allocation.
  auto raw_ptr = manager.Malloc(1 << 11);
  ASSERT_EQ(-1, manager.PoolIdxForPtr(static_cast<uint8_t*>(raw_ptr)));

  auto* ptr = manager.Malloc((1 << 9) - manager.kHeaderSize);
  ASSERT_EQ(ptr, manager.buffer_[0] + manager.kHeaderSize);
  // The 1-th bucket should not be empty
  ASSERT_TRUE(manager.buckets_[1]);
  for (int i = 2; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(manager.buckets_[i]);
  }

  ptr = manager.Malloc((1 << 9) - manager.kHeaderSize);
  ASSERT_EQ(ptr, manager.buffer_[0] + (1 << 9) + manager.kHeaderSize);

  for (int i = 1; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(manager.buckets_[i]);
  }

  // OOM
  ASSERT_FALSE(manager.Malloc(1 << 2));
}

TEST(BuddyManager, Malloc1) {
  BuddyManager manager(2, 10, 9);

  auto ShowBuckets = [&] {
    for (int i = 0; i < manager.num_buckets_; i++) {
      LOG(INFO) << i << " " << ListGetSizeSlow(manager.buckets_[i])
                << " bucket_size " << manager.BucketSize(i);
    }

    for (int v : std::vector<int>({0, 1, 3, 7, 8, 15})) {
      LOG(INFO) << "node is flip " << v << " " << manager.is_splits_[0].Tell(v);
    }
  };

  auto CheckIsSplits = [&](const std::unordered_set<int>& nodes) {
    for (int i = 0; i < manager.is_splits_[0].num_bytes(); i++) {
      if (nodes.count(i)) {
        if (!manager.is_splits_[0].Tell(i)) {
          LOG(ERROR) << "node " << i << " != true";
        }
        EXPECT_TRUE(manager.is_splits_[0].Tell(i));
      } else {
        if (manager.is_splits_[0].Tell(i)) {
          LOG(ERROR) << "node " << i << " != false";
        }
        EXPECT_FALSE(manager.is_splits_[0].Tell(i));
      }
    }
  };

  LOG(INFO) << "allocate " << (1 << 5) << " ----------------------------";
  // need extra 1<<3 if Header included, so an 1<<6 block is needed.
  // The node 15 is used.
  auto* node_15 = manager.Malloc(1 << 5);
  ShowBuckets();
  CheckIsSplits({0, 1, 3, 7});
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1);  // 1<<9
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[2]), 1);  // 1<<8
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[3]), 1);  // 1<<7
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[4]), 1);  // 1<<6

  LOG(INFO) << "allocate " << (1 << 5) << " ----------------------------";
  // Alloc another 1<<6 block, the node 16 is used, so the whole node 7 is used.
  auto* node_16 = manager.Malloc(1 << 5);
  ShowBuckets();
  CheckIsSplits({0, 1, 3});
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1);  // 1<<9
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[2]), 1);  // 1<<8
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[3]), 1);  // 1<<7
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[4]), 0);  // 1<<6

  // Alloc another 1<<6 block, the node 17 is used.
  auto* node_17 = manager.Malloc(1 << 5);
  ShowBuckets();
  CheckIsSplits({0, 1, 8});  // node 3 is used, so not split
  // Alloc another 1<<6 block, the node 18 is used, and the whole node 8 is
  // used.
  auto* node_18 = manager.Malloc(1 << 5);
  ShowBuckets();
  CheckIsSplits({0, 1});  // node 3 is used, so not split

  // Free node 17
  manager.Free(node_17);
  ShowBuckets();
  CheckIsSplits({0, 1, 8});
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1);  // 1<<9
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[2]), 1);  // 1<<8
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[3]), 0);  // 1<<7
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[4]), 1);  // 1<<6, node17

  // Free node 15
  manager.Free(node_15);
  ShowBuckets();
  CheckIsSplits({0, 1, 7, 8});
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1);  // 1<<9
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[2]), 1);  // 1<<8
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[3]), 0);  // 1<<7
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[4]),
            2);  // 1<<6, node17 and node15

  // Free node 16, that block should merge with the block of node15, and make
  // the entire node7 free.
  manager.Free(node_16);
  ShowBuckets();
  CheckIsSplits({0, 1, 3, 8});  // node7 is free, and make node3 split
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1);  // 1<<9
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[2]), 1);  // 1<<8
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[3]), 1);  // 1<<7
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[4]),
            1);  // 1<<6, node17 and node15

  auto* node_4 = manager.Malloc((1 << 8) - manager.kHeaderSize);
  ShowBuckets();
  CheckIsSplits({0, 3, 8});

  // Alloc node5 and node6, and that will make node2 used.
  auto* node_5 = manager.Malloc((1 << 8) - manager.kHeaderSize);
  auto* node_6 = manager.Malloc((1 << 8) - manager.kHeaderSize);
  ShowBuckets();
  CheckIsSplits({3, 8});
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 0);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 0);  // 1<<9, node2 is used.
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
