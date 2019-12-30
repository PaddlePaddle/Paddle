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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace detail {

void MemoryBlock::Init(void* data, Type type, size_t index, size_t size,
                       MemoryBlock* left_buddy, MemoryBlock* right_buddy) {
  this->data_ = data;
  this->type_ = type;
  this->index_ = index;
  this->size_ = size;
  this->left_buddy_ = left_buddy;
  this->right_buddy_ = right_buddy;
}

bool MemoryBlock::Split(size_t size, MemoryBlockPool* pool) {
  if (this->size_ == size) return false;

  // make sure the split fits
  PADDLE_ENFORCE_GT(this->size_, size);

  // find the position of the split
  void* right_partition = reinterpret_cast<uint8_t*>(this->data_) + size;

  // Add the new block as a buddy
  // Write the metadata for the new block
  auto original_rb = this->right_buddy_;

  this->right_buddy_ = pool->Get();
  this->right_buddy_->Init(right_partition, FREE_CHUNK, this->index_,
                           this->size_ - size, this, original_rb);

  this->size_ = size;

  // Write metadata for the new block's right buddy
  if (original_rb != nullptr) {
    original_rb->left_buddy_ = this->right_buddy_;
  }
  return true;
}

void MemoryBlock::Merge(MemoryBlock* block, MemoryBlockPool* pool) {
  // only free blocks can be merged
  PADDLE_ENFORCE_EQ(this->type_, FREE_CHUNK);
  PADDLE_ENFORCE_EQ(block->type_, FREE_CHUNK);

  // link this->buddy's buddy
  this->right_buddy_ = block->right_buddy_;

  // link buddy's buddy -> this
  if (this->right_buddy_ != nullptr) {
    this->right_buddy_->left_buddy_ = this;
  }

  this->size_ += block->size_;
  pool->Push(block);
}

MemoryBlockPool::~MemoryBlockPool() {
  for (auto ptr : raw_pointers_) delete ptr;
}

MemoryBlock* MemoryBlockPool::Get() {
  // if no free MemoryBlock is available, then refill the pool
  if (pool_.empty()) {
    raw_pointers_.emplace_back(new MemoryBlock[inc_size_]);
    for (auto i = 0; i < inc_size_; ++i)
      pool_.emplace(raw_pointers_.back() + i);
  }

  auto block = pool_.front();
  pool_.pop();
  return block;
}

void MemoryBlockPool::Push(MemoryBlock* block) { pool_.emplace(block); }

}  // namespace detail
}  // namespace memory
}  // namespace paddle
