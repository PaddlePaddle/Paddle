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

DEFINE_int32(pile_allocator_init_memory_size_in_mb, 200,
             "Initial memory pool size for PileAllocator");
DEFINE_int32(pile_allocator_realloc_memory_size_in_mb, 100,
             "Initial memory pool size for PileAllocator");

namespace paddle {
namespace memory {
namespace allocation {

void BuddySystem::Free(void* p) {
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

void* BuddySystem::MallocImpl(size_t request, size_t pool_idx) {
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

}  // namespace allocation
}  // namespace memory
}  // namespace paddle
