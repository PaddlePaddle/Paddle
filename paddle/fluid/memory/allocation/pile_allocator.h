#include <gtest/gtest_prod.h>
#include <list>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/memory/allocation/allocator.h"

namespace paddle {
namespace memory {
namespace allocation {

// Stupid data structure, why the allocated memory should return with an
// unique_ptr of Allocation?
// This design adds overhead of CPU allocation.
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

// Stack with Set.
template <typename T>
struct StackSet {
  bool count(T t) { return map_.count(t); }

  T top() const { return stack_.front(); }

  size_t size() const { return stack_.size(); }

  void push(T v) {
    stack_.push_front(v);
    map_.emplace(v, stack_.begin());
  }

  void pop() {
    auto v = stack_.front();
    map_.erase(v);
    stack_.pop_front();
  }

  void pop(T v) {
    auto it = map_.find(v);
    CHECK(it != map_.end());
    stack_.erase(it->second);
    map_.erase(it);
  }

  bool empty() const { return stack_.empty(); }

 private:
  std::list<T> stack_;
  std::unordered_map<T, typename std::list<T>::iterator> map_;
};

/*
 * The resource for buddy system, we split the resource from the buddy system
 * for better reuse for mulitple ones.
 * The same underlying memory pool can be reused that, each buddy system can
 * allcate memory based on the same memory pool, regardless of others, so the
 * same memory block can be allcated by more than one buddy systems, on
 * condition that the buddy systems are used sequentially and individually.
 */
struct BuddyResource {
  using byte_t = uint8_t;
  using bucket_t = StackSet<byte_t*>;

  BuddyResource(int max_mem_log_size, int min_mem_log_size,
                int realloc_mem_log_size, int num_piled = 1)
      : min_mem_log_size_(min_mem_log_size),
        min_mem_size_(1 << min_mem_log_size),
        max_mem_log_size_(max_mem_log_size),
        max_mem_size_(1 << max_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1),
        realloc_mem_log_size_(realloc_mem_log_size),
        realloc_mem_size_(1 << realloc_mem_log_size),
        num_piled_(num_piled) {
    buckets_.resize(num_piled_);
    is_splits_.resize(num_piled_);

    Init();
  }

  ~BuddyResource() {
    for (auto* v : buffer_) {
      delete[] v;
    }
  }

  std::vector<byte_t*>& buffer() { return buffer_; }
  std::vector<bucket_t>& buckets(int idx) { return buckets_[idx]; }
  std::vector<BitSet>& is_splits(int idx) { return is_splits_[idx]; }

  byte_t* SystemAllocate(size_t size) {
    auto* x = new byte_t[size];
    PADDLE_ENFORCE(x, "system OOM! allocate %d bytes failed.", max_mem_size_);
    return x;
  }

  void ReallocMemoryBuffer() {
    LOG(INFO) << "Realloc one memory block";
    PADDLE_ENFORCE_GT(buffer_.size(), 0UL,
                      "should allocate the initial memory pool first.");
    buffer_.push_back(SystemAllocate(realloc_mem_size_));
    PushInitialBucket(max_mem_log_size() - realloc_mem_log_size(),
                      buffer_.back());

    // Add one is_splits for each pile system.
    for (auto& is_splits : is_splits_) {
      is_splits.emplace_back();
      is_splits.back().Resize(
          1 << (realloc_mem_log_size() - min_mem_log_size() + 1));
    }
  }

#define GET_MEMBER(field__) \
  int field__() const { return field__##_; }
  GET_MEMBER(min_mem_log_size);
  GET_MEMBER(max_mem_log_size);
  GET_MEMBER(min_mem_size);
  GET_MEMBER(max_mem_size);
  GET_MEMBER(realloc_mem_log_size);
  GET_MEMBER(realloc_mem_size);
  GET_MEMBER(num_piled);
  GET_MEMBER(num_buckets);
#undef GET_MEMBER

 protected:
  void Init() {
    // Init is_splits
    for (auto& is_splits : is_splits_) {
      PADDLE_ENFORCE(is_splits.empty());
      is_splits.emplace_back();
      is_splits.front().Resize(1 << num_buckets_);
    }

    // Init buckets
    for (auto& buckets : buckets_) {
      buckets.resize(num_buckets());
    }

    AllocInitialMemoryBuffer();
  }

  void AllocInitialMemoryBuffer() {
    PADDLE_ENFORCE(buffer_.empty());
    buffer_.push_back(SystemAllocate(max_mem_size_));
    PushInitialBucket(0, buffer_.front());
  }

  // Push the bucket to all the pile systems' free buckets.
  void PushInitialBucket(size_t bucket, byte_t* ptr) {
    for (auto& buckets : buckets_) {
      buckets[bucket].push(ptr);
    }
  }

 private:
  const int min_mem_log_size_;
  const int max_mem_log_size_;
  const int min_mem_size_;
  const int max_mem_size_;
  const uint32_t num_buckets_;
  const int realloc_mem_log_size_;
  const int realloc_mem_size_;

  // number of the piled system that share the same memory buffer.
  const int num_piled_;
  // The buffer will shared by all the piled system.
  std::vector<byte_t*> buffer_;
  // All the memory pools shares the same buckets. The outer vector is for piled
  // systems, the innser vector is for different size of buddies.
  std::vector<std::vector<bucket_t>> buckets_;
  // State represents is_split, one bit for each node. The outer vector is for
  // multiple pile system, the inner vector is for multiple buffer.
  std::vector<std::vector<BitSet>> is_splits_;
};

static std::shared_ptr<BuddyResource> CreateBuddyResource(
    int max_mem_log_size, int min_mem_log_size, int realloc_mem_log_size,
    int num_piled = 1) {
  return std::make_shared<BuddyResource>(max_mem_log_size, min_mem_log_size,
                                         realloc_mem_log_size, num_piled);
}

/**
 * Memory model:
 * The underlying memory is an buffer allocated by system, the buckets is some
 * descriptions wraps it.
 *
 * There are several data structure on it:
 * - ListNode: the double-link list node, which help to maintain the free list
 * for buckets.
 * - Bucket: a free list of ListNodes.
 * - request: an size_t value which indicate the memory needed. For each
 *            allocated memory block, there is an 8-byte header ahead which
 *            store the `request`.
 * - memory pool: the memory buffer allocated from system, the BuddyManager will
 *                re-manage the allocation on it.
 *
 * For each free bucket, the memory is
 *   | ListNode |  ... rest memory of this bucket |
 * For each used bucket, the memory is
 *   | Request | ... rest memory of this bucket |
 *
 * Reallocation:
 * If the initial memory pool is full-filled, the extra memory pool is needed.
 * This will triger the system allocation, and put the memory pool reallcated to
 * the free buckets.
 */
struct BuddySystem {
  using byte_t = uint8_t;
  using bucket_t = typename BuddyResource::bucket_t;

  BuddySystem(const std::shared_ptr<BuddyResource>& resource,
              size_t pile_idx = 0, bool allow_realloc = true)
      : min_mem_log_size_(resource->min_mem_log_size()),
        max_mem_log_size_(resource->max_mem_log_size()),
        min_mem_size_(resource->min_mem_size()),
        max_mem_size_(resource->max_mem_size()),
        realloc_mem_log_size_(resource->realloc_mem_log_size()),
        realloc_mem_size_(resource->realloc_mem_size()),
        num_buckets_(resource->num_buckets()),
        resource_(resource),
        allow_realloc_(allow_realloc),
        buffer_(&resource->buffer()) {
    is_splits_ = &resource->is_splits(pile_idx);
    buckets_ = &resource->buckets(pile_idx);

    PADDLE_ENFORCE_NOT_NULL(is_splits_);
    PADDLE_ENFORCE_NOT_NULL(buffer_);
    PADDLE_ENFORCE_NOT_NULL(buckets_);
    PADDLE_ENFORCE(!buffer_->empty(), "BuddyResource should be inited first.");
    PADDLE_ENFORCE(!buckets_->front().empty());
  }

  BuddySystem(size_t min_mem_log_size, size_t max_mem_log_size,
              size_t realloc_mem_log_size, bool allow_realloc = true)
      : min_mem_log_size_(min_mem_log_size),
        max_mem_log_size_(max_mem_log_size),
        min_mem_size_(1 << min_mem_log_size),
        max_mem_size_(1 << max_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1),
        realloc_mem_log_size_(realloc_mem_log_size),
        realloc_mem_size_(1 << realloc_mem_log_size),
        allow_realloc_{allow_realloc} {
    resource_ = CreateBuddyResource(max_mem_log_size, min_mem_log_size,
                                    realloc_mem_log_size, 1);
    is_splits_ = &resource_->is_splits(0);
    buffer_ = &resource_->buffer();
    buckets_ = &resource_->buckets(0);

    PADDLE_ENFORCE_NOT_NULL(is_splits_);
    PADDLE_ENFORCE_NOT_NULL(buffer_);
    PADDLE_ENFORCE_NOT_NULL(buckets_);
    PADDLE_ENFORCE(!buffer_->empty(), "BuddyResource should be inited first.");
    PADDLE_ENFORCE(!buckets_->front().empty());
  }

  void* Malloc(size_t request) {
    if (request > max_mem_size_) {
      LOG(INFO) << "allocating raw large memory chunk";
      return resource_->SystemAllocate(request);
    }
    auto* first_try = MallocImpl(request, 0);
    if (first_try) return first_try;

    if (!allow_realloc_) return nullptr;
    resource_->ReallocMemoryBuffer();
    return MallocImpl(request, 0);
  }

  void Free(void* p) {
    auto* ptr = static_cast<byte_t*>(p);
    // Ignore null.
    if (!ptr) return;

    size_t request = ptr2request_[ptr];
    size_t bucket = BucketForRequest(request);
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
      auto* list_node = PtrForNode(brother, bucket, pool_idx);
      PopBucket(bucket, list_node);
      // Jump to parent
      node = (node - 1) / 2;
      bucket--;
    }
    PushBucket(bucket, PtrForNode(node, bucket, pool_idx));
  }

 protected:
  void* MallocImpl(size_t request, size_t pool_idx) {
    if (request > max_mem_size_) {
      LOG(ERROR) << "OOM";
      return nullptr;
    }
    PADDLE_ENFORCE(!buffer_->empty());

    // We should reserve the memory for Header of request.
    size_t origin_bucket = BucketForRequest(request);
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
                << is_splits_->at(pool_idx).Tell(index);
        bucket++;
      }

      ptr2request_[ptr] = request;
      return ptr;
    }
    return nullptr;
  }

  byte_t* buffer(int idx) { return buffer_->at(idx); }

  BitSet& is_splits(int idx) { return is_splits_->at(idx); }

  bucket_t& buckets(int idx) {
    PADDLE_ENFORCE_LT(idx, buckets_->size());
    return buckets_->at(idx);
  }

  // TODO(Superjomn) Optimize the performance here.
  int PoolIdxForPtr(byte_t* ptr) {
    if (buffer_->front() <= ptr && buffer_->front() + max_mem_size_ > ptr)
      return 0;
    for (int i = 1; i < buffer_->size(); i++) {
      if (buffer_->at(i) <= ptr && buffer_->at(i) + realloc_mem_size_ > ptr)
        return i;
    }
    return -1;
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
    is_splits_->at(pool_idx).Flip(index);
  }

  bool IsParentSplit(size_t index, size_t pool_idx) {
    index = (index - 1) / 2;
    return is_splits_->at(pool_idx).Tell(index);
  }

  byte_t* PopBucket(size_t bucket) {
    auto& the_bucket = buckets(bucket);
    if (the_bucket.empty()) return nullptr;
    auto* ptr = the_bucket.top();
    the_bucket.pop();
    return ptr;
  }

  void PopBucket(size_t bucket, byte_t* ptr) { buckets(bucket).pop(ptr); }

  void PushBucket(size_t bucket, byte_t* ptr) { buckets(bucket).push(ptr); }

  size_t GetBrotherNode(size_t node) { return ((node - 1) ^ 1) + 1; }

  /** Split the bucket, and push the right child bucket to the free list */
  void SplitBucket(size_t bucket, byte_t* ptr, size_t sub_bucket_size,
                   size_t pool_idx) {
    size_t sub_bucket = BucketForRequest(sub_bucket_size);
    size_t index = NodeForPtr(ptr, bucket, pool_idx);
    LOG(INFO) << "index " << index;
    PADDLE_ENFORCE_NOT_NULL(is_splits_);
    PADDLE_ENFORCE_GT(is_splits_->size(), pool_idx);
    is_splits_->at(pool_idx).Flip(index);
    PushBucket(sub_bucket, ptr + sub_bucket_size);
  }

  /** Get the node's index for a memory address */
  size_t NodeForPtr(byte_t* ptr, size_t bucket, size_t pool_idx) {
    auto offset = (ptr - buffer_->at(pool_idx)) >> (max_mem_log_size_ - bucket);
    return offset + (1 << bucket) - 1;
  }

  /**
   * Get the memory address of a node.
   */
  byte_t* PtrForNode(size_t index, size_t bucket, size_t pool_idx) {
    return buffer_->at(pool_idx) +
           ((index - (1 << bucket) + 1) << (max_mem_log_size_ - bucket));
  }

 private:
  // All the memory pools shares the same buckets.
  std::vector<bucket_t>* buckets_{nullptr};
  // State represents is_split, one bit for each node.
  std::vector<BitSet>* is_splits_{nullptr};

  // map from allocated memory address to the requested memory size.
  std::unordered_map<byte_t*, uint32_t> ptr2request_;

  const int min_mem_log_size_;
  const int max_mem_log_size_;
  const size_t min_mem_size_;
  const size_t max_mem_size_;
  const uint32_t num_buckets_;
  const int realloc_mem_log_size_;
  const int realloc_mem_size_;

  const bool allow_realloc_;

  std::vector<byte_t*>* buffer_{nullptr};

  std::shared_ptr<BuddyResource> resource_;

  FRIEND_TEST(BuddySystem, test1);
  FRIEND_TEST(BuddySystem, BucketForRequest);
  FRIEND_TEST(BuddySystem, BucketSize);
  FRIEND_TEST(BuddySystem, TestFirstBucket);
  FRIEND_TEST(BuddySystem, PushBucket);
  FRIEND_TEST(BuddySystem, PopBucket);
  FRIEND_TEST(BuddySystem, GetBrotherNode);
  FRIEND_TEST(BuddySystem, SplitBucket);
  FRIEND_TEST(BuddySystem, NodeForPtr);
  FRIEND_TEST(BuddySystem, PtrForNode);
  FRIEND_TEST(BuddySystem, Malloc);
  FRIEND_TEST(BuddySystem, Malloc1);
  FRIEND_TEST(BuddySystem, realloc);
};

/** \brief An (best-fit)allocator with memory share enabled.
 * This allocator offers the sequential sharing of temporary workspace.
 *
 * The example scenerios:
 *   We have N models, each has an average memory size W for weight and T for
 * temporary variables, and all these models are used in sequential. By default,
 * these models will occupy N*(W+T) size of memory.
 *   By using PileAllocator, we can share the temporary variable memory space
 * for all the models, and in tatal these model will only take N*W+T size of
 * memory. That performance is remarkable when T is large.
 *
 */
class PileAllocator : public Allocator {
 public:
  using byte_t = char;

  struct MemoryOption {
    MemoryOption(size_t max_memory_size, size_t min_memory_size,
                 size_t realloc_memory_size)
        : max_memory_log_size(std::floor(std::log(max_memory_size))),
          min_memory_log_size(std::floor(std::log(min_memory_size))),
          realloc_memory_log_size(std::floor(std::log(realloc_memory_size))),
          max_memory_size(1 << max_memory_log_size),
          min_memory_size(1 << min_memory_log_size),
          realloc_memory_size(1 << realloc_memory_log_size) {}

    const size_t max_memory_log_size;
    const size_t min_memory_log_size;
    const size_t realloc_memory_log_size;

    const size_t max_memory_size;
    const size_t min_memory_size;
    const size_t realloc_memory_size;
  };

  PileAllocator(int num_pile, const MemoryOption& meta_memory,
                const MemoryOption& pile_memory) {
    meta_system_.reset(new BuddySystem(meta_memory.max_memory_log_size,
                                       meta_memory.min_memory_log_size,
                                       meta_memory.realloc_memory_log_size, 1));
    auto pile_resource = CreateBuddyResource(
        pile_memory.max_memory_log_size, pile_memory.min_memory_log_size,
        pile_memory.realloc_memory_log_size, num_pile);

    for (int i = 0; i < num_pile; i++) {
      meta_system_.reset(new BuddySystem(pile_resource, i));
    }
  }

  void SetPileIdx(int idx) { pile_idx_ = idx; }

  // TODO(Superjomn) TODO(px) re-consider the allocation interface.
  void* Allocate(size_t size, Allocator::Attr attr = kDefault) {
    if (pile_idx_ == -1) {
      return meta_system_->Malloc(size);
    }
    return pile_systems_.at(pile_idx_)->Malloc(size);
  }

  bool IsAllocThreadSafe() const override { return false; }

 private:
  std::unique_ptr<BuddySystem> meta_system_;
  std::vector<std::unique_ptr<BuddySystem>> pile_systems_;
  int pile_idx_{-1};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
