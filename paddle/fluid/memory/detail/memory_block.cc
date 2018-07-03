/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/memory/detail/memory_block.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace memory {
namespace detail {

void MemoryBlock::init(MetadataCache* cache, Type t, size_t index, size_t size,
                       void* left_buddy, void* right_buddy) {
  cache->save(
      this, MemoryBlock::Desc(t, index, size - sizeof(MemoryBlock::Desc), size,
                              static_cast<MemoryBlock*>(left_buddy),
                              static_cast<MemoryBlock*>(right_buddy)));
}

MemoryBlock::Type MemoryBlock::type(const MetadataCache& cache) const {
  return cache.load(this).type;
}

size_t MemoryBlock::size(const MetadataCache& cache) const {
  return cache.load(this).size;
}

size_t MemoryBlock::index(const MetadataCache& cache) const {
  return cache.load(this).index;
}

size_t MemoryBlock::total_size(const MetadataCache& cache) const {
  return cache.load(this).total_size;
}

bool MemoryBlock::has_left_buddy(const MetadataCache& cache) const {
  return left_buddy(cache) != nullptr;
}

bool MemoryBlock::has_right_buddy(const MetadataCache& cache) const {
  return right_buddy(cache) != nullptr;
}

MemoryBlock* MemoryBlock::left_buddy(const MetadataCache& cache) const {
  return cache.load(this).left_buddy;
}

MemoryBlock* MemoryBlock::right_buddy(const MetadataCache& cache) const {
  return cache.load(this).right_buddy;
}

void MemoryBlock::split(MetadataCache* cache, size_t size) {
  // make sure the split fits
  PADDLE_ASSERT(total_size(*cache) >= size);

  // bail out if there is no room for another partition
  if (total_size(*cache) - size <= sizeof(MemoryBlock::Desc)) {
    return;
  }

  // find the position of the split
  void* right_partition = reinterpret_cast<uint8_t*>(this) + size;

  size_t remaining_size = total_size(*cache) - size;

  // Add the new block as a buddy
  auto metadata = cache->load(this);

  // Write the metadata for the new block
  auto new_block_right_buddy = metadata.right_buddy;

  cache->save(static_cast<MemoryBlock*>(right_partition),
              MemoryBlock::Desc(FREE_CHUNK, index(*cache),
                                remaining_size - sizeof(MemoryBlock::Desc),
                                remaining_size, this, new_block_right_buddy));

  metadata.right_buddy = static_cast<MemoryBlock*>(right_partition);
  metadata.size = size - sizeof(MemoryBlock::Desc);
  metadata.total_size = size;

  cache->save(this, metadata);

  // Write metadata for the new block's right buddy
  if (new_block_right_buddy != nullptr) {
    auto buddy_metadata = cache->load(new_block_right_buddy);

    buddy_metadata.left_buddy = static_cast<MemoryBlock*>(right_partition);

    cache->save(new_block_right_buddy, buddy_metadata);
  }
}

void MemoryBlock::merge(MetadataCache* cache, MemoryBlock* right_buddy) {
  // only free blocks can be merged
  PADDLE_ASSERT(type(*cache) == FREE_CHUNK);
  PADDLE_ASSERT(right_buddy->type(*cache) == FREE_CHUNK);

  auto metadata = cache->load(this);

  // link this->buddy's buddy
  metadata.right_buddy = right_buddy->right_buddy(*cache);

  // link buddy's buddy -> this
  if (metadata.right_buddy != nullptr) {
    auto buddy_metadata = cache->load(metadata.right_buddy);

    buddy_metadata.left_buddy = this;

    cache->save(metadata.right_buddy, buddy_metadata);
  }

  metadata.size += right_buddy->total_size(*cache);
  metadata.total_size += right_buddy->total_size(*cache);

  cache->save(this, metadata);
  cache->save(right_buddy,
              MemoryBlock::Desc(INVALID_CHUNK, 0, 0, 0, nullptr, nullptr));
}

void MemoryBlock::mark_as_free(MetadataCache* cache) {
  // check for double free or corruption
  PADDLE_ASSERT(type(*cache) != FREE_CHUNK);
  PADDLE_ASSERT(type(*cache) != INVALID_CHUNK);
  set_type(cache, FREE_CHUNK);
}

void MemoryBlock::set_type(MetadataCache* cache, Type t) {
  auto metadata = cache->load(this);
  metadata.type = t;
  cache->save(this, metadata);
}

void* MemoryBlock::data() const {
  return const_cast<MemoryBlock::Desc*>(
             reinterpret_cast<const MemoryBlock::Desc*>(this)) +
         1;
}

MemoryBlock* MemoryBlock::metadata() const {
  return const_cast<MemoryBlock*>(reinterpret_cast<const MemoryBlock*>(
      reinterpret_cast<const MemoryBlock::Desc*>(this) - 1));
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
