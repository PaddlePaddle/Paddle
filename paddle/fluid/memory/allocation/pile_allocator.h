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

#include <gflags/gflags.h>
#include <gtest/gtest_prod.h>
#include <list>
#include <mutex>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/memory/allocation/allocator.h"
#include "paddle/fluid/memory/detail/system_allocator.h"

DECLARE_int32(pile_allocator_init_memory_size_in_mb);
DECLARE_int32(pile_allocator_realloc_memory_size_in_mb);

namespace paddle {
namespace memory {
namespace allocation {

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
// A low-performance implementation.
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
  enum class Place { kCPU = 0, kGPU };

  BuddyResource(int max_mem_log_size, int min_mem_log_size,
                int realloc_mem_log_size, int num_piled = 1,
                Place place = Place::kCPU)
      : min_mem_log_size_(min_mem_log_size),
        min_mem_size_(1 << min_mem_log_size),
        max_mem_log_size_(max_mem_log_size),
        max_mem_size_(1 << max_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1),
        realloc_mem_log_size_(realloc_mem_log_size),
        realloc_mem_size_(1 << realloc_mem_log_size),
        num_piled_(num_piled),
        place_(place) {
    PADDLE_ENFORCE_LE(min_mem_log_size, 35);
    PADDLE_ENFORCE_LE(max_mem_log_size, 35);
    PADDLE_ENFORCE_LE(realloc_mem_log_size, 35);
    PADDLE_ENFORCE_LE(num_buckets_, 10000);
    PADDLE_ENFORCE_LE(num_piled, 10);

    buckets_.resize(num_piled_);
    is_splits_.resize(num_piled_);

    Init();
  }

  ~BuddyResource() {
    for (auto* v : buffer_) {
      SystemFree(v);
    }
  }

  std::vector<byte_t*>& buffer() { return buffer_; }
  std::vector<bucket_t>& buckets(int idx) { return buckets_[idx]; }
  std::vector<BitSet>& is_splits(int idx) { return is_splits_[idx]; }
  size_t total_memory_pool_size() const { return total_memory_pool_size_; }

  void* SystemAllocate(size_t size) {
    total_memory_pool_size_ += size;
    switch (place_) {
      case Place::kCPU:
        return CpuSystemAllocate(size);
        break;
#ifdef PADDLE_WITH_CUDA
      case Place::kGPU:
        return GpuSystemAllocate(size);
        break;
#endif
      default:
        LOG(ERROR) << "wrong place get";
        return nullptr;
    }
  }

  void SystemFree(void* ptr) {
    switch (place_) {
      case Place::kCPU:
        CpuSystemFree(ptr);
        break;
#ifdef PADDLE_WITH_CUDA
      case Place::kGPU:
        GpuSystemFree(ptr);
        break;
#endif
      default:
        LOG(ERROR) << "wrong place get";
    }
  }

  // We implement simple system allocators for CPU and GPU devices, for the
  // existing SystemAllocators interface is trivial.
  static void* CpuSystemAllocate(size_t size) { return new byte_t[size]; }
  static void CpuSystemFree(void* ptr) { delete[] static_cast<byte_t*>(ptr); }

#ifdef PADDLE_WITH_CUDA
  static void* GpuSystemAllocate(size_t size) {
    byte_t* ptr;
    PADDLE_ENFORCE(cudaMalloc(&ptr, size));
    return ptr;
  }
  static void GpuSystemFree(void* ptr) { PADDLE_ENFORCE(cudaFree(ptr)); }
#endif

  void ReallocMemoryBuffer() {
    LOG(INFO) << "Realloc one memory block";
    PADDLE_ENFORCE_GT(buffer_.size(), 0UL,
                      "should allocate the initial memory pool first.");
    buffer_.push_back(
        reinterpret_cast<byte_t*>(SystemAllocate(realloc_mem_size_)));
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
    LOG(INFO) << "buckets_size " << buckets_.size();
    LOG(INFO) << "num_buckets " << num_buckets_;
    for (auto& buckets : buckets_) {
      buckets.resize(num_buckets());
    }

    AllocInitialMemoryBuffer();
  }

  void AllocInitialMemoryBuffer() {
    PADDLE_ENFORCE(buffer_.empty());
    buffer_.push_back(reinterpret_cast<byte_t*>(SystemAllocate(max_mem_size_)));
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
  const int min_mem_size_;
  const int max_mem_log_size_;
  const int max_mem_size_;
  const uint32_t num_buckets_;
  const int realloc_mem_log_size_;
  const int realloc_mem_size_;

  // number of the piled system that share the same memory buffer.
  const int num_piled_;
  // The buffer will shared by all the piled system.
  std::vector<byte_t*> buffer_;

  size_t total_memory_pool_size_{0};

  // All the memory pools shares the same buckets. The outer vector is for
  // piled
  // systems, the inner vector is for different size of buddies.
  std::vector<std::vector<bucket_t>> buckets_;
  // State represents is_split, one bit for each node. The outer vector is for
  // multiple pile system, the inner vector is for multiple buffer.
  std::vector<std::vector<BitSet>> is_splits_;

  Place place_;
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
 * Reallocation:
 * If the initial memory pool is full-filled, the extra memory pool is needed.
 * This will triger the system allocation, and put the memory pool reallcated
 * to
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
        allow_realloc_(allow_realloc),
        resource_(resource),
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
        realloc_mem_log_size_(realloc_mem_log_size),
        realloc_mem_size_(1 << realloc_mem_log_size),
        num_buckets_(max_mem_log_size - min_mem_log_size + 1),
        allow_realloc_{allow_realloc} {
    PADDLE_ENFORCE_LE(max_mem_log_size, 35,
                      "memory should be set lower than 34G");
    PADDLE_ENFORCE_LE(min_mem_log_size, 35,
                      "memory should be set lower than 34G");
    PADDLE_ENFORCE_LE(realloc_mem_log_size, 35,
                      "memory should be set lower than 34G");

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
    std::lock_guard<std::mutex> lk(mut_);
    if (request > max_mem_size_) {
      return resource_->SystemAllocate(request);
    }

    auto* first_try = MallocImpl(request, 0);
    if (first_try) {
      return first_try;
    }

    if (!allow_realloc_) return nullptr;

    resource_->ReallocMemoryBuffer();
    return MallocImpl(request, 0);
  }

  void Free(void* p);

  size_t request_memory_size() const { return request_memory_size_; }
  size_t buddy_free_size() const { return buddy_free_size_; }
  float frag_ratio() {
    return static_cast<double>(request_memory_size_) /
           (resource_->total_memory_pool_size() - buddy_free_size_);
  }

 protected:
  void* MallocImpl(size_t request, size_t pool_idx);

  byte_t* buffer(int idx) { return buffer_->at(idx); }

  BitSet& is_splits(int idx) { return is_splits_->at(idx); }

  bucket_t& buckets(int idx) {
    PADDLE_ENFORCE_LT(idx, static_cast<int>(buckets_->size()));
    return buckets_->at(idx);
  }

  // TODO(Superjomn) Optimize the performance here.
  int PoolIdxForPtr(byte_t* ptr) {
    if (buffer_->front() <= ptr && buffer_->front() + max_mem_size_ > ptr)
      return 0;
    for (size_t i = 1; i < buffer_->size(); i++) {
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

  void PopBucket(size_t bucket, byte_t* ptr) {
    buddy_free_size_ -= BucketSize(bucket);
    buckets(bucket).pop(ptr);
  }

  void PushBucket(size_t bucket, byte_t* ptr) {
    buddy_free_size_ += BucketSize(bucket);
    buckets(bucket).push(ptr);
  }

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
  const int min_mem_log_size_;
  const int max_mem_log_size_;
  const size_t min_mem_size_;
  const size_t max_mem_size_;
  const int realloc_mem_log_size_;
  const int realloc_mem_size_;

  const uint32_t num_buckets_;
  const bool allow_realloc_;
  std::mutex mut_;

  std::shared_ptr<BuddyResource> resource_;

  // All the memory pools shares the same buckets.
  std::vector<bucket_t>* buckets_{nullptr};
  // State represents is_split, one bit for each node.
  std::vector<BitSet>* is_splits_{nullptr};

  // map from allocated memory address to the requested memory size.
  std::unordered_map<byte_t*, uint32_t> ptr2request_;

  std::vector<byte_t*>* buffer_{nullptr};

  // The sum of the exact memory size user requested.
  size_t request_memory_size_{0};
  // The sum of the free buddies in the free buckets.
  size_t buddy_free_size_{0};

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
 * temporary variables, and all these models are used in sequential. By
 * default,
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
                 size_t realloc_memory_size);

    const size_t max_memory_log_size;
    const size_t min_memory_log_size;
    const size_t realloc_memory_log_size;

    const size_t max_memory_size;
    const size_t min_memory_size;
    const size_t realloc_memory_size;
  };

  PileAllocator(int num_pile, const MemoryOption& meta_memory,
                const MemoryOption& pile_memory) {
    meta_system_.reset(new BuddySystem(meta_memory.min_memory_log_size,
                                       meta_memory.max_memory_log_size,
                                       meta_memory.realloc_memory_log_size, 1));
    auto pile_resource = CreateBuddyResource(
        pile_memory.max_memory_log_size, pile_memory.min_memory_log_size,
        pile_memory.realloc_memory_log_size, num_pile);

    for (int i = 0; i < num_pile; i++) {
      meta_system_.reset(new BuddySystem(pile_resource, i));
    }
  }

  float frag_ratio(int idx) {
    if (idx == -1) {
      return meta_system_->frag_ratio();
    }
    return pile_systems_.at(idx)->frag_ratio();
  }

  void SetPileIdx(int idx) {
    PADDLE_ENFORCE_LT(idx, 1UL + pile_systems_.size());
    pile_idx_ = idx;
  }

  void Free(Allocation* allocation) override {
    RawFree(allocation->ptr());
    delete allocation;
  }

  void* RawAllocate(size_t size) {
    switch (pile_idx_) {
      case -1:
        return meta_system_->Malloc(size);
      default:
        return pile_systems_.at(pile_idx_)->Malloc(size);
    }
  }

  void RawFree(void* ptr) {
    switch (pile_idx_) {
      case -1:
        meta_system_->Free(ptr);
        break;
      default:
        pile_systems_.at(pile_idx_)->Free(ptr);
    }
  }

  bool IsAllocThreadSafe() const override { return false; }

 private:
  // It should be low-performance when allocating the Allocation. Each time it
  // allocates, two malloc will be called.
  // TODO(px) Re-consider the overall allocator interface design.
  Allocation* AllocateImpl(size_t size,
                           Allocator::Attr attr = kDefault) override {
    auto* p = new Allocation(RawAllocate(size), size, platform::CPUPlace());
    p->set_allocator(this);
    return p;
  }

  std::unique_ptr<BuddySystem> meta_system_;
  std::vector<std::unique_ptr<BuddySystem>> pile_systems_;
  int pile_idx_{-1};
};

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
