#include "paddle/memory/detail/memory_block.h"
#include "paddle/platform/assert.h"

namespace paddle {
namespace memory {
namespace detail {

void MemoryBlock::init(MetadataCache& cache, Type t, size_t index, size_t size,
                       void* left_buddy, void* right_buddy) {
  cache.store(this,
              MemoryBlockMetadata(t, index, size - overhead(), size,
                                  static_cast<MemoryBlock*>(left_buddy),
                                  static_cast<MemoryBlock*>(right_buddy)));
}

MemoryBlock::Type MemoryBlock::type(MetadataCache& cache) const {
  return cache.load(this).type;
}

size_t MemoryBlock::size(MetadataCache& cache) const {
  return cache.load(this).size;
}

size_t MemoryBlock::total_size(MetadataCache& cache) const {
  return cache.load(this).total_size;
}

MemoryBlock* MemoryBlock::left_buddy(MetadataCache& cache) const {
  return cache.load(this).left_buddy;
}

MemoryBlock* MemoryBlock::right_buddy(MetadataCache& cache) const {
  return cache.load(this).right_buddy;
}

void MemoryBlock::split(MetadataCache& cache, size_t size) {
  // make sure the split fits
  assert(total_size(cache) >= size);

  // bail out if there is no room for another partition
  if (total_size(cache) - size <= overhead()) {
    return;
  }

  // find the position of the split
  void* right_partition = reinterpret_cast<uint8_t*>(this) + size;

  size_t remaining_size = total_size(cache) - size;

  // Add the new block as a buddy
  auto metadata = cache.load(this);

  // Write the metadata for the new block
  auto new_block_right_buddy = metadata.right_buddy;

  cache.store(static_cast<MemoryBlock*>(right_partition),
              MemoryBlockMetadata(FREE_MEMORY, index(cache),
                                  remaining_size - overhead(), remaining_size,
                                  this, new_block_right_buddy));

  metadata.right_buddy = static_cast<MemoryBlock*>(right_partition);
  metadata.size = size - overhead();
  metadata.total_size = size;

  cache.store(this, metadata);

  // Write metadata for the new block's right buddy
  if (new_block_right_buddy != nullptr) {
    auto buddy_metadata = cache.load(new_block_right_buddy);

    buddy_metadata.left_buddy = static_cast<MemoryBlock*>(right_partition);

    cache.store(new_block_right_buddy, buddy_metadata);
  }
}

void MemoryBlock::merge(MetadataCache& cache, MemoryBlock* right_buddy) {
  // only free blocks can be merged
  assert(type(cache) == FREE_MEMORY);
  assert(right_buddy->type(cache) == FREE_MEMORY);

  auto metadata = cache.load(this);

  // link this->buddy's buddy
  metadata.right_buddy = right_buddy->right_buddy(cache);

  // link buddy's buddy -> this
  if (metadata.right_buddy != nullptr) {
    auto buddy_metadata = cache.load(metadata.right_buddy);

    buddy_metadata.left_buddy = this;

    cache.store(metadata.right_buddy, buddy_metadata);
  }

  metadata.size += right_buddy->total_size(cache);
  metadata.total_size += right_buddy->total_size(cache);

  cache.store(this, metadata);
  cache.store(right_buddy,
              MemoryBlockMetadata(INVALID_MEMORY, 0, 0, 0, nullptr, nullptr));
}

void MemoryBlock::mark_as_free(MetadataCache& cache) {
  // check for double free or corruption
  assert(type(cache) != FREE_MEMORY);
  assert(type(cache) != INVALID_MEMORY);

  set_type(cache, FREE_MEMORY);
}

void MemoryBlock::set_type(MetadataCache& cache, Type t) {
  auto metadata = cache.load(this);

  metadata.type = t;

  cache.store(this, metadata);
}

bool MemoryBlock::has_left_buddy(MetadataCache& cache) const {
  return left_buddy(cache) != nullptr;
}

bool MemoryBlock::has_right_buddy(MetadataCache& cache) const {
  return right_buddy(cache) != nullptr;
}

size_t MemoryBlock::index(MetadataCache& cache) const {
  return cache.load(this).index;
}

void* MemoryBlock::data() const {
  return const_cast<MemoryBlockMetadata*>(
             reinterpret_cast<const MemoryBlockMetadata*>(this)) +
         1;
}

MemoryBlock* MemoryBlock::metadata() const {
  return const_cast<MemoryBlock*>(reinterpret_cast<const MemoryBlock*>(
      reinterpret_cast<const MemoryBlockMetadata*>(this) - 1));
}

}  // detail
}  // memory
}  // paddle
