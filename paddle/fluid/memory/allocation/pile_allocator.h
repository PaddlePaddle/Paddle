#include <gtest/gtest_prod.h>
#include <list>
#include <stack>
#include <unordered_map>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

class PileAllocation : public Allocation {};

struct BitSet {
  using byte_t = unsigned char;

  BitSet() = default;
  BitSet(size_t num_bits) { Resize(num_bits); }

  void Resize(size_t num_bits) {
    buffer_.resize(std::ceil(num_bits / 8.), 0);
    size_ = num_bits;
  }

  void Set(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx - 8 * (idx / 8);
    byte ^= 1 << offset;
  }

  void UnSet(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx - 8 * (idx / 8);

    byte &= ~(1 << offset);
  }

  void Flip(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx - 8 * (idx / 8);
    byte ^= 1 << offset;
  }

  bool Tell(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx - 8 * (idx / 8);
    return byte & 1 << idx;
  }

  size_t num_bytes() const { return buffer_.size(); }

  size_t size() const { return size_; }

 private:
  std::vector<byte_t> buffer_;
  size_t size_{0};
};

struct ListNode {
  ListNode* pre;
  ListNode* next;

  ListNode() { Init(); }

  void Init() {
    pre = this;
    next = this;
  }

  void Append(ListNode* other) {
    auto* the_next = this->next;
    other->pre = this;
    this->next = other;
    other->next = the_next;
    the_next->pre = other;
  }

  void Remove() {
    auto* preone = pre;
    preone->next = this->next;
    this->next->pre = preone;
  }
};

size_t ListGetSizeSlow(ListNode* head) {
  if (!head) return 0;
  auto* p = head;

  size_t res = 0;
  while (p) {
    p = p->next;
    ++res;
    if (p == head) break;
  }
  return res;
}

/**
 * Memory model:
 * The underlying memory is an buffer allocated by system, the buckets is some
 * descriptions wraps it.
 *
 * There are several data structure on it:
 * - ListNode: the double-link list node, which help to maintain the free list
 * for buckets.
 * - Bucket: a free list
 * - Request: an size_t value which indicate the memory used.
 *
 * For each free bucket, the memory is
 *   | ListNode |  ... rest memory of this bucket |
 * For each used bucket, the memory is
 *   | Request | ... rest memory of this bucket |
 */
struct BuddyManager {
  using byte_t = unsigned char;
  // An integer is needed to store the size of request. We use a 8-byte header
  // to store this integer to keep memory aligned by 8.
  const uint32_t kHeaderSize{sizeof(size_t)};

  BuddyManager(size_t min_mem_log_size, size_t max_mem_log_size)
      : min_mem_log_size_(min_mem_log_size),
        max_mem_log_size_(max_mem_log_size),
        min_mem_size_(1 << min_mem_log_size),
        max_mem_size_(1 << max_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1) {
    Init();
  }

  void* Malloc(size_t request) {
    if (request + kHeaderSize > max_mem_size_) {
      LOG(ERROR) << "OOM";
      return nullptr;
    }
    PADDLE_ENFORCE_NOT_NULL(buffer_);

    // We should reserve the memory for Header of request.
    size_t origin_bucket = BucketForRequest(request + kHeaderSize);
    size_t bucket = origin_bucket;

    while (bucket + 1 != 0) {
      // If no free list in current bucket, go to bigger bucket and try.
      // time complexity: O(logN)
      byte_t* ptr = reinterpret_cast<byte_t*>(PopBucket(bucket));

      if (!ptr) {
        --bucket;
        continue;
      }

      while (bucket < origin_bucket) {
        size_t size = BucketSize(bucket);
        SplitBucket(bucket, ptr, size / 2);
        size_t index = NodeForPtr(ptr, bucket);

        LOG(INFO) << "split bucket " << bucket << " node " << index
                  << " fliped " << is_splits_.Tell(index);
        bucket++;
      }

      *reinterpret_cast<size_t*>(ptr) = request;
      return ptr + kHeaderSize;
    }
    return nullptr;
  }

  void Free(void* p) {
    byte_t* ptr = static_cast<byte_t*>(p);
    // Ignore null.
    if (!ptr) return;

    ptr = ptr - kHeaderSize;
    size_t request = *reinterpret_cast<size_t*>(ptr);
    size_t bucket = BucketForRequest(request + kHeaderSize);
    size_t node = NodeForPtr(ptr, bucket);

    while (node != 0) {
      FlipParentIsSplit(node);
      LOG(INFO) << "parent is split " << IsParentSplit(node) << " node "
                << (node - 1) / 2 << " bucket_size " << BucketSize(bucket - 1);
      // The bucket is used, no need to merge.
      if (IsParentSplit(node)) break;

      // Here, the bucket is not used, remove the bucket from free list.
      size_t brother = GetBrotherNode(node);
      auto* listnode = PtrForNode(brother, bucket);
      reinterpret_cast<ListNode*>(listnode)->Remove();
      if (buckets_[bucket] == reinterpret_cast<ListNode*>(listnode))
        buckets_[bucket] = nullptr;
      // Jump to parent
      node = (node - 1) / 2;
      bucket--;
    }
    PushBucket(bucket, reinterpret_cast<ListNode*>(PtrForNode(node, bucket)));
  }

  ~BuddyManager() { delete[] buffer_; }

 protected:
  /** The tree is always rooted at the largest bucket, and grow if needed. It
   * will grow leefs until hit the bucket limit.
   */
  void Init() {
    buckets_.resize(num_buckets_, nullptr);
    labels_.resize(1 << num_buckets_, 0);

    InitIsSplits();
    AllocFirstBucket();
  }

  /**
   * Allocate the largest bucket, and add it to the free list.
   */
  void AllocFirstBucket() {
    PADDLE_ENFORCE_GE(max_mem_size_, sizeof(ListNode));
    buffer_ = SystemAllocate(max_mem_size_);
    PushBucket(0, reinterpret_cast<ListNode*>(buffer_));
  }

  void InitIsSplits() { is_splits_.Resize(1 << num_buckets_); }

  /** Get the minest bucket that satisfy the request.
   */
  size_t BucketForRequest(size_t request) {
    size_t bucket_idx = num_buckets_ - 1;
    auto size = min_mem_size_;
    while (size < request) {
      bucket_idx--;
      size *= 2;
    }
    return bucket_idx;
  }

  size_t BucketSize(size_t bucket) {
    return (1 << (min_mem_log_size_ + num_buckets_ - bucket - 1));
  }

  void FlipParentIsSplit(size_t index) {
    if (index == 0) return;
    index = (index - 1) / 2;
    is_splits_.Flip(index);
  }

  bool IsParentSplit(size_t index) {
    index = (index - 1) / 2;
    return is_splits_.Tell(index);
  }

  ListNode* PopBucket(size_t bucket) {
    auto* head = buckets_[bucket];
    if (!head) return nullptr;

    ListNode* res;
    if (head->next != head) {
      // more than one nodes, pop the second node.
      res = head->next;
      res->Remove();
    } else {
      // Just one node
      res = head;
      buckets_[bucket] = nullptr;
    }

    return res;
  }

  void PushBucket(size_t bucket, ListNode* ptr) {
    auto& head = buckets_[bucket];
    if (!head) {
      head = ptr;
      head->Init();
    } else {
      head->Append(ptr);
    }
  }

  size_t GetBrotherNode(size_t node) { return ((node - 1) ^ 1) + 1; }

  /** Split the bucket, and push the right child bucket to the free list */
  void SplitBucket(size_t bucket, byte_t* ptr, size_t sub_bucket_size) {
    size_t sub_bucket = BucketForRequest(sub_bucket_size);
    size_t index = NodeForPtr(ptr, bucket);
    is_splits_.Flip(index);
    PushBucket(sub_bucket, reinterpret_cast<ListNode*>(ptr + sub_bucket_size));
  }

  /** Get the node's index for a memory address */
  size_t NodeForPtr(byte_t* ptr, size_t bucket) {
    auto offset = (ptr - buffer_) >> (max_mem_log_size_ - bucket);
    return offset + (1 << bucket) - 1;
  }

  /**
   * Get the memory address of a node.
   */
  byte_t* PtrForNode(size_t index, size_t bucket) {
    return buffer_ +
           ((index - (1 << bucket) + 1) << (max_mem_log_size_ - bucket));
  }

  byte_t* SystemAllocate(size_t size) {
    auto* x = new byte_t[size];
    PADDLE_ENFORCE(x, "system OOM! allocate %d bytes failed.", max_mem_size_);
    return x;
  }

 private:
  std::vector<ListNode*> buckets_;
  std::vector<uint8_t> labels_;

  // State represents is_split, one bit for each node.
  BitSet is_splits_;

  const int min_mem_log_size_;
  const int max_mem_log_size_;
  const size_t min_mem_size_;
  const size_t max_mem_size_;
  const uint32_t num_buckets_;

  byte_t* buffer_{nullptr};

  FRIEND_TEST(BuddyManager, test1);
  FRIEND_TEST(BuddyManager, BucketForRequest);
  FRIEND_TEST(BuddyManager, BucketSize);
  FRIEND_TEST(BuddyManager, TestFirstBucket);
  FRIEND_TEST(BuddyManager, PushBucket);
  FRIEND_TEST(BuddyManager, PopBucket);
  FRIEND_TEST(BuddyManager, GetBrotherNode);
  FRIEND_TEST(BuddyManager, SplitBucket);
  FRIEND_TEST(BuddyManager, NodeForPtr);
  FRIEND_TEST(BuddyManager, PtrForNode);
  FRIEND_TEST(BuddyManager, Malloc);
  FRIEND_TEST(BuddyManager, Malloc1);
};

/** \brief An (best-fit)allocator with memory share enabled.
 *
 * This is an special best-fit allocator, it supports pile memory share.
 * About pile memory share: the memory chunk be shared by multiple consumers.
 * We will color the chunks with several labels, and determine which label can
 * be shared while others cannot be shared.
 */
class PileAllocator : public Allocator {
 public:
  using byte_t = char;

  PileAllocator(size_t max_memory_size = 1 << 20,
                size_t basic_cluster_size = 1 << 10, size_t page_size = 64,
                size_t init_mem_size = 1 << 20, size_t inc_size = 1 << 20)
      : kMaxMemorySize(max_memory_size),
        kBasicClusterSize(basic_cluster_size),
        kPageSize(page_size),
        kInitMemSize(init_mem_size),
        kIncSize(inc_size) {}

  AllocationPtr Allocate(size_t size, Allocator::Attr attr = kDefault) {
    if (size > kMaxMemorySize) return nullptr;
  }

  bool IsAllocThreadSafe() const override { return false; }

  Allocation* AllocateImpl(size_t size, Allocator::Attr attr) override {
    return nullptr;
  }

 protected:
  void* SystemAllocate(size_t size) {
    auto* x = new byte_t[size];
    PADDLE_ENFORCE(x, "system OOM! allocate %d bytes failed.", kInitMemSize);
    return x;
  }

 private:
  const size_t kMaxMemorySize;
  const size_t kBasicClusterSize;
  const uint32_t kPageSize{64};
  const size_t kInitMemSize;
  const size_t kIncSize;
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
