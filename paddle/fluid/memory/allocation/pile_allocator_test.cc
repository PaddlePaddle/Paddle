#include "paddle/fluid/memory/allocation/pile_allocator.h"
#include <gtest/gtest.h>

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
}

TEST(BuddyManager, test1) {
  BuddyManager manager(4, 10);
  LOG(INFO) << "min mem: " << manager.min_mem_size_ << " "
            << manager.min_mem_log_size_;
  LOG(INFO) << "max mem: " << manager.max_mem_size_ << " "
            << manager.max_mem_log_size_;
}

TEST(BuddyManager, BucketForRequest) {
  BuddyManager manager(4, 10);
  LOG(INFO) << "num buckets: " << manager.num_buckets_;
  ASSERT_EQ(manager.BucketForRequest(1 << 10), 0);
  ASSERT_EQ(manager.BucketForRequest(1 << 9), 1);
  for (int i = 0; i < 1 << 8; i++) {
    ASSERT_EQ(manager.BucketForRequest((1 << 9) - i), 1);
  }
}

TEST(BuddyManager, BucketSize) {
  BuddyManager manager(4, 10);
  ASSERT_EQ(manager.BucketSize(0), 1 << 10);
  ASSERT_EQ(manager.BucketSize(1), 1 << 9);
}

TEST(BuddyManager, TestFirstBucket) {
  BuddyManager manager(4, 10);
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
  ASSERT_EQ(reinterpret_cast<uint8_t*>(res), manager.buffer_);
  ASSERT_EQ(res->next, res);
  ASSERT_EQ(res->pre, res);
  // Pop if there is no remaining buckets.
  ASSERT_FALSE(manager.PopBucket(0));
}

TEST(BuddyManager, PushBucket) {
  BuddyManager manager(4, 10);

  // Push to the first bucket.
  ASSERT_FALSE(manager.buckets_[1]);

  manager.PushBucket(
      1, reinterpret_cast<ListNode*>(manager.buffer_ + sizeof(ListNode)));
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1UL);
  ASSERT_EQ(manager.buckets_[1]->next, manager.buckets_[1]);
  ASSERT_EQ(manager.buckets_[1]->pre, manager.buckets_[1]);

  // There are 1<<10 nodes for the 0th bucket.
  for (int i = 2; i < 7; i++) {
    manager.PushBucket(
        1, reinterpret_cast<ListNode*>(manager.buffer_ + i * sizeof(ListNode)));
    ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), i);
  }
}

TEST(BuddyManager, PopBucket) {
  BuddyManager manager(4, 10);
  for (int i = 0; i < 4; i++) {
    manager.PushBucket(
        0, reinterpret_cast<ListNode*>(manager.buffer_ + i * sizeof(ListNode)));
  }

  for (int i = 0; i < 4; i++) {
    manager.PopBucket(0);
    ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 4 - i - 1);
  }
}

TEST(BuddyManager, GetBrotherNode) {
  BuddyManager manager(4, 10);
  ASSERT_EQ(manager.GetBrotherNode(1), 2);
  ASSERT_EQ(manager.GetBrotherNode(3), 4);
  ASSERT_EQ(manager.GetBrotherNode(4), 3);
  ASSERT_EQ(manager.GetBrotherNode(5), 6);
  ASSERT_EQ(manager.GetBrotherNode(6), 5);
  ASSERT_EQ(manager.GetBrotherNode(7), 8);
  ASSERT_EQ(manager.GetBrotherNode(8), 7);
}

TEST(BuddyManager, SplitBucket) {
  BuddyManager manager(4, 10);
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[0]), 1UL);
  manager.SplitBucket(0, manager.buffer_, (1 << 10) / 2);
  // The large bucket will not poped by default.
  ASSERT_EQ(ListGetSizeSlow(manager.buckets_[1]), 1UL);
  ASSERT_EQ(reinterpret_cast<uint8_t*>(manager.buckets_[1]),
            manager.buffer_ + (1 << 10) / 2);

  ASSERT_TRUE(manager.is_splits_.Tell(0));
}

TEST(BuddyManager, NodeForPtr) {
  BuddyManager manager(4, 10);
  ASSERT_EQ(
      manager.NodeForPtr(manager.buffer_, manager.BucketForRequest(1 << 10)),
      0UL);

  ASSERT_EQ(
      manager.NodeForPtr(manager.buffer_, manager.BucketForRequest(1 << 9)),
      1UL);
  ASSERT_EQ(manager.NodeForPtr(manager.buffer_ + (1 << 9),
                               manager.BucketForRequest(1 << 9)),
            2UL);
  for (int i = 0; i < 4; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_ + i * (1 << 8),
                                 manager.BucketForRequest(1 << 8)),
              2UL + i + 1);
  }

  for (int i = 0; i < 4 * 2; i++) {
    ASSERT_EQ(manager.NodeForPtr(manager.buffer_ + i * (1 << 7),
                                 manager.BucketForRequest(1 << 7)),
              6UL + i + 1);
  }
}

TEST(BuddyManager, PtrForNode) {
  BuddyManager manager(4, 10);
  // same start point with different buddy size.
  ASSERT_EQ(manager.PtrForNode(0, 0), manager.buffer_);
  ASSERT_EQ(manager.PtrForNode(1, 1), manager.buffer_);
  ASSERT_EQ(manager.PtrForNode(2, 1), manager.buffer_ + (1 << 9));
  ASSERT_EQ(manager.PtrForNode(3, 2), manager.buffer_);
  ASSERT_EQ(manager.PtrForNode(4, 2), manager.buffer_ + 1 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(5, 2), manager.buffer_ + 2 * (1 << 8));
  ASSERT_EQ(manager.PtrForNode(6, 2), manager.buffer_ + 3 * (1 << 8));
}

TEST(BuddyManager, Malloc) {
  BuddyManager manager(4, 10);
  // OOM
  ASSERT_FALSE(manager.Malloc(1 << 11));

  auto* ptr = manager.Malloc((1 << 9) - manager.kHeaderSize);
  ASSERT_EQ(ptr, manager.buffer_ + manager.kHeaderSize);
  // The 1-th bucket should not be empty
  ASSERT_TRUE(manager.buckets_[1]);
  for (int i = 2; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(manager.buckets_[i]);
  }

  ptr = manager.Malloc((1 << 9) - manager.kHeaderSize);
  ASSERT_EQ(ptr, manager.buffer_ + (1 << 9) + manager.kHeaderSize);

  for (int i = 1; i < manager.num_buckets_; i++) {
    ASSERT_FALSE(manager.buckets_[i]);
  }

  // OOM
  ASSERT_FALSE(manager.Malloc(1 << 2));
}

TEST(BuddyManager, Malloc1) {
  BuddyManager manager(2, 10);

  auto ShowBuckets = [&] {
    for (int i = 0; i < manager.num_buckets_; i++) {
      LOG(INFO) << i << " " << ListGetSizeSlow(manager.buckets_[i])
                << " bucket_size " << manager.BucketSize(i);
    }

    /*
    LOG(INFO) << "is_splits ";
    for (int i = 0; i < manager.is_splits_.size(); i++) {
      LOG(INFO) << i << " " << manager.is_splits_.Tell(i);
    }
     */

    for (int v : std::vector<int>({0, 1, 3, 7, 15})) {
      LOG(INFO) << "node is flip " << v << " " << manager.is_splits_.Tell(v);
    }
  };

  LOG(INFO) << "allocate " << (1 << 4) << " ----------------------------";
  // need 1<<3 if Header included
  auto* ptr = manager.Malloc(1 << 4);
  // the 0th bucket will be splitted, and an 1th bucket will be added.
  // the 1th bucket's buddy will be split to one 2th, one 3th, one 4th, one 5th
  // buddies.
  ShowBuckets();

  LOG(INFO) << "free " << (1 << 4) << " ----------------------------";
  manager.Free(ptr);
  ShowBuckets();

  manager.Malloc(1 << 4);
  ShowBuckets();

  manager.Malloc(1 << 4);
  ShowBuckets();
}

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
