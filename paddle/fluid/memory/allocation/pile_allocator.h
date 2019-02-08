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
    byte |= (1 << offset);
  }

  void UnSet(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx - 8 * (idx / 8);

    byte &= ~(1 << offset);
  }

  void Flip(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx % 8;
    byte ^= (1 << offset);
  }

  bool Tell(size_t idx) {
    auto& byte = buffer_[idx / 8];
    int offset = idx % 8;
    return byte & (1 << offset);
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
  using byte_t = uint8_t;
  // An integer is needed to store the size of request. We use a 8-byte header
  // to store this integer to keep memory aligned by 8.
  const uint32_t kHeaderSize{sizeof(size_t)};

  BuddyManager(size_t min_mem_log_size, size_t max_mem_log_size,
               size_t realloc_mem_log_size, bool allow_realloc = true)
      : min_mem_log_size_(min_mem_log_size),
        max_mem_log_size_(max_mem_log_size),
        min_mem_size_(1 << min_mem_log_size),
        max_mem_size_(1 << max_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1),
        realloc_mem_log_size_(realloc_mem_log_size),
        realloc_mem_size_(1 << realloc_mem_log_size),
        allow_realloc_{allow_realloc} {
    Init();
  }

  void* Malloc(size_t request) {
    if (request > max_mem_size_) {
      LOG(INFO) << "allocating raw large memory chunk";
      return SystemAllocate(request);
    }
    auto* first_try = MallocImpl(request, 0);
    if (first_try) return first_try;

    if (!allow_realloc_) return nullptr;
    ReallocMemoryBuffer();
    return MallocImpl(request, 0);
  }

  // Label the allocated memory block.
  void MarkAllocated(byte_t* ptr, bool flag) {
    size_t request = *reinterpret_cast<size_t*>(ptr - kHeaderSize);
    size_t pool_idx = PoolIdxForPtr(ptr);
    auto node = NodeForPtr(ptr, BucketForRequest(request), pool_idx);
    if (flag) {
      labels_[pool_idx].Set(node);
    } else {
      labels_[pool_idx].UnSet(node);
    }
  }

  void Free(void* p) {
    byte_t* ptr = static_cast<byte_t*>(p);
    // Ignore null.
    if (!ptr) return;

    ptr = ptr - kHeaderSize;
    size_t request = *reinterpret_cast<size_t*>(ptr);
    size_t bucket = BucketForRequest(request + kHeaderSize);
    int pool_idx = PoolIdxForPtr(ptr);
    // System allocated.
    if (pool_idx < 0) {
      delete[] ptr;
      return;
    }

    size_t node = NodeForPtr(ptr, bucket, pool_idx);

    while (node != 0) {
      FlipParentIsSplit(node, pool_idx);
      VLOG(3) << "parent is split " << IsParentSplit(node, pool_idx) << " node "
              << (node - 1) / 2 << " bucket_size " << BucketSize(bucket - 1);
      // The bucket is used, no need to merge.
      if (IsParentSplit(node, pool_idx)) break;

      // Here, the bucket is not used, remove the bucket from free list.
      size_t brother = GetBrotherNode(node);
      auto* listnode =
          reinterpret_cast<ListNode*>(PtrForNode(brother, bucket, pool_idx));
      PopBucket(bucket, listnode);
      if (buckets_[bucket] == reinterpret_cast<ListNode*>(listnode))
        buckets_[bucket] = nullptr;
      // Jump to parent
      node = (node - 1) / 2;
      bucket--;
    }
    PushBucket(bucket,
               reinterpret_cast<ListNode*>(PtrForNode(node, bucket, pool_idx)));
  }

  ~BuddyManager() {
    for (auto* v : buffer_) {
      delete[] v;
    }
  }

 protected:
  /** The tree is always rooted at the largest bucket, and grow if needed. It
   * will grow leefs until hit the bucket limit.
   */
  void Init() {
    buckets_.resize(num_buckets_, nullptr);

    AllocFirstBucket();
    labels_.front().Resize(1 << num_buckets_);
    is_splits_.front().Resize(1 << num_buckets_);
  }

  void* MallocImpl(size_t request, size_t pool_idx) {
    if (request + kHeaderSize > max_mem_size_) {
      LOG(ERROR) << "OOM";
      return nullptr;
    }
    PADDLE_ENFORCE(!buffer_.empty());

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

      size_t pool_idx = PoolIdxForPtr(ptr);
      size_t index = NodeForPtr(ptr, bucket, pool_idx);
      if (index != 0) FlipParentIsSplit(index, pool_idx);

      while (bucket < origin_bucket) {
        size_t size = BucketSize(bucket);
        SplitBucket(bucket, ptr, size / 2, pool_idx);
        size_t index = NodeForPtr(ptr, bucket, pool_idx);

        VLOG(3) << "split bucket " << bucket << " node " << index << " fliped "
                << is_splits_[pool_idx].Tell(index);
        bucket++;
      }

      *reinterpret_cast<size_t*>(ptr) = request;
      return ptr + kHeaderSize;
    }
    return nullptr;
  }

  // TODO(Superjomn) Optimize the performance here.
  int PoolIdxForPtr(byte_t* ptr) {
    if (buffer_.front() <= ptr && buffer_.front() + max_mem_size_ > ptr)
      return 0;
    for (int i = 1; i < buffer_.size(); i++) {
      if (buffer_[i] <= ptr && buffer_[i] + realloc_mem_size_ > ptr) return i;
    }
    return -1;
  }

  /**
   * Allocate the largest bucket, and add it to the free list.
   */
  void AllocFirstBucket() {
    PADDLE_ENFORCE_GE(max_mem_size_, sizeof(ListNode));
    AllocInitialMemoryBuffer();
    PushBucket(0, reinterpret_cast<ListNode*>(buffer_.front()));

    labels_.emplace_back();
    is_splits_.emplace_back();
  }

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

  void FlipParentIsSplit(size_t index, size_t pool_idx) {
    if (index == 0) return;
    index = (index - 1) / 2;
    is_splits_[pool_idx].Flip(index);
  }

  bool IsParentSplit(size_t index, size_t pool_idx) {
    index = (index - 1) / 2;
    return is_splits_[pool_idx].Tell(index);
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

  void PopBucket(size_t bucket, ListNode* ptr) {
    if (ptr->next == ptr) {
      buckets_[bucket] = nullptr;
    } else {
      ptr->Remove();
    }
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
  void SplitBucket(size_t bucket, byte_t* ptr, size_t sub_bucket_size,
                   size_t pool_idx) {
    size_t sub_bucket = BucketForRequest(sub_bucket_size);
    size_t index = NodeForPtr(ptr, bucket, pool_idx);
    is_splits_[pool_idx].Flip(index);
    PushBucket(sub_bucket, reinterpret_cast<ListNode*>(ptr + sub_bucket_size));
  }

  /** Get the node's index for a memory address */
  size_t NodeForPtr(byte_t* ptr, size_t bucket, size_t pool_idx) {
    auto offset = (ptr - buffer_[pool_idx]) >> (max_mem_log_size_ - bucket);
    return offset + (1 << bucket) - 1;
  }

  /**
   * Get the memory address of a node.
   */
  byte_t* PtrForNode(size_t index, size_t bucket, size_t pool_idx) {
    return buffer_[pool_idx] +
           ((index - (1 << bucket) + 1) << (max_mem_log_size_ - bucket));
  }

  void AllocInitialMemoryBuffer() {
    buffer_.push_back(SystemAllocate(max_mem_size_));
  }

  void ReallocMemoryBuffer() {
    LOG(INFO) << "Realloc one memory block";
    PADDLE_ENFORCE_GT(buffer_.size(), 0UL,
                      "should allocate the initial memory pool first.");
    buffer_.push_back(SystemAllocate(realloc_mem_size_));
    PushBucket(BucketForRequest(realloc_mem_size_),
               reinterpret_cast<ListNode*>(buffer_.back()));
    labels_.emplace_back();
    is_splits_.emplace_back();

    labels_.back().Resize(realloc_mem_size_/min_mem_size_);
    is_splits_.back().Resize(realloc_mem_size_/min_mem_size_);
  }

  byte_t* SystemAllocate(size_t size) {
    auto* x = new byte_t[size];
    PADDLE_ENFORCE(x, "system OOM! allocate %d bytes failed.", max_mem_size_);
    return x;
  }

 private:
  // All the memory pools shares the same buckets.
  std::vector<ListNode*> buckets_;
  // State represents is_split, one bit for each node.
  std::vector<BitSet> is_splits_;
  // Label which mark the nodes, we use a bitset to support an 0/1 label.
  std::vector<BitSet> labels_;

  const int min_mem_log_size_;
  const int max_mem_log_size_;
  const size_t min_mem_size_;
  const size_t max_mem_size_;
  const uint32_t num_buckets_;
  const int realloc_mem_log_size_;
  const int realloc_mem_size_;

  const bool allow_realloc_;

  std::vector<byte_t*> buffer_;

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
  FRIEND_TEST(BuddyManager, realloc);
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
    return nullptr;
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
