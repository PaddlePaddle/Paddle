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

#include "paddle/phi/core/memory/allocation/memory_block.h"

#include "paddle/phi/core/enforce.h"

namespace paddle::memory::detail {

void MemoryBlock::Init(MetadataCache* cache,
                       Type t,
                       size_t index,
                       size_t size,
                       void* left_buddy,
                       void* right_buddy) {
  cache->Save(this,
              MemoryBlock::Desc(t,
                                index,
                                size - sizeof(MemoryBlock::Desc),
                                size,
                                static_cast<MemoryBlock*>(left_buddy),
                                static_cast<MemoryBlock*>(right_buddy)));
}

MemoryBlock* MemoryBlock::GetLeftBuddy(MetadataCache* cache) {
  return cache->LoadDesc(this)->left_buddy;
}

MemoryBlock* MemoryBlock::GetRightBuddy(MetadataCache* cache) {
  return cache->LoadDesc(this)->right_buddy;
}

void MemoryBlock::Split(MetadataCache* cache,
                        size_t size,
                        size_t extra_padding_size) {
  auto desc = cache->LoadDesc(this);
  // make sure the split fits
  PADDLE_ENFORCE_GE(desc->total_size,
                    size,
                    common::errors::InvalidArgument(
                        "The size of memory block (%d) to split is "
                        "not larger than size of request memory (%d)",
                        desc->total_size,
                        size));

  size_t pay_load_size = sizeof(MemoryBlock::Desc) + extra_padding_size;

  // bail out if there is no room for another partition
  if (desc->total_size - size <= pay_load_size) {
    return;
  }

  // find the position of the split
  void* right_partition = reinterpret_cast<uint8_t*>(this) + size;

  size_t remaining_size = desc->total_size - size;

  // Add the new block as a buddy
  // Write the metadata for the new block
  auto new_block_right_buddy = desc->right_buddy;

  cache->Save(static_cast<MemoryBlock*>(right_partition),
              MemoryBlock::Desc(FREE_CHUNK,
                                desc->index,
                                remaining_size - pay_load_size,
                                remaining_size,
                                this,
                                new_block_right_buddy));

  desc->right_buddy = static_cast<MemoryBlock*>(right_partition);
  desc->size = size - pay_load_size;
  desc->total_size = size;

  desc->UpdateGuards();

  // Write metadata for the new block's right buddy
  if (new_block_right_buddy != nullptr) {
    auto buddy_desc = cache->LoadDesc(new_block_right_buddy);

    buddy_desc->left_buddy = static_cast<MemoryBlock*>(right_partition);
    buddy_desc->UpdateGuards();
  }
}

void MemoryBlock::Merge(MetadataCache* cache, MemoryBlock* right_buddy) {
  // only free blocks can be merged
  auto desc = cache->LoadDesc(this);
  auto rb_desc = cache->LoadDesc(right_buddy);
  PADDLE_ENFORCE_EQ(desc->type,
                    FREE_CHUNK,
                    common::errors::PreconditionNotMet(
                        "The destination chunk to merge is not free"));
  PADDLE_ENFORCE_EQ(rb_desc->type,
                    FREE_CHUNK,
                    common::errors::PreconditionNotMet(
                        "The source chunk to merge is not free"));

  // link this->buddy's buddy
  desc->right_buddy = rb_desc->right_buddy;

  // link buddy's buddy -> this
  if (desc->right_buddy != nullptr) {
    auto buddy_metadata = cache->LoadDesc(desc->right_buddy);

    buddy_metadata->left_buddy = this;
    buddy_metadata->UpdateGuards();
  }

  desc->size += rb_desc->total_size;
  desc->total_size += rb_desc->total_size;

  desc->UpdateGuards();

  cache->Save(right_buddy,
              MemoryBlock::Desc(INVALID_CHUNK, 0, 0, 0, nullptr, nullptr));
}

void MemoryBlock::MarkAsFree(MetadataCache* cache) {
  // check for double free or corruption
  auto desc = cache->LoadDesc(this);
  PADDLE_ENFORCE_NE(desc->type,
                    FREE_CHUNK,
                    common::errors::PreconditionNotMet(
                        "The chunk to mark as free is free already"));
  PADDLE_ENFORCE_NE(desc->type,
                    INVALID_CHUNK,
                    common::errors::PreconditionNotMet(
                        "The chunk to mark as free is invalid"));
  desc->type = FREE_CHUNK;
  desc->UpdateGuards();
}

void* MemoryBlock::Data() const {
  return const_cast<MemoryBlock::Desc*>(
             reinterpret_cast<const MemoryBlock::Desc*>(this)) +
         1;
}

MemoryBlock* MemoryBlock::Metadata() const {
  return const_cast<MemoryBlock*>(reinterpret_cast<const MemoryBlock*>(
      reinterpret_cast<const MemoryBlock::Desc*>(this) - 1));
}

}  // namespace paddle::memory::detail
