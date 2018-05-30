// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/memory/detail/memory_block2.h"
#include <functional>

namespace paddle {
namespace memory {
namespace detail {
namespace {

template <class T>
inline void hash_combine(std::uint64_t* seed, const T& v) {
  std::hash<T> hasher;
  (*seed) ^= hasher(v) + 0x9e3779b9 + ((*seed) << 6) + ((*seed) >> 2);
}

inline uint64_t hash(const MemoryBlock& block, uint64_t initial_seed) {
  uint64_t seed = initial_seed;

  hash_combine(&seed, static_cast<uint64_t>(block.state_));
  hash_combine(&seed, block.raw_ptr_);
  hash_combine(&seed, block.size_);
  hash_combine(&seed, block.left_buddy_);
  hash_combine(&seed, block.right_buddy_);

  return seed;
}

}  // namespace

MemoryBlock::MemoryBlock()
    : state_(State::INVALID_BLOCK),
      raw_ptr_(nullptr),
      size_(0UL),
      left_buddy_(nullptr),
      right_buddy_(nullptr) {
  UpdateGuards();
}

MemoryBlock::MemoryBlock(void* block, uint64_t size)
    : state_(State::FREE_BLOCK),
      raw_ptr_(block),
      size_(size),
      left_buddy_(nullptr),
      right_buddy_(nullptr) {
  UpdateGuards();
}

MemoryBlock::~MemoryBlock() {
  // ensure the left, right buddy is released before.
  PADDLE_ENFORCE(left_buddy_ == nullptr, "left_buddy error");
  PADDLE_ENFORCE(right_buddy_ == nullptr, "right_buddy error");
  PADDLE_ENFORCE(CheckGuards() == true, "memory block error");
  Reset();
}

void MemoryBlock::Reset() {
  state_ = State::INVALID_BLOCK;
  raw_ptr_ = nullptr;
  size_ = 0;
  guard_begin_ = guard_end_ = 0;
}

void MemoryBlock::SplitToBuddy() {
  left_buddy_.reset(new MemoryBlock(raw_ptr_, size / 2));
  right_buddy_.reset(new MemoryBlock(raw_ptr_ + (size / 2), size / 2));
  Reset();
}

void MemoryBlock::MergeFromBuddy() {
  bool valid =
      (left_buddy_ != nullptr) && (left_buddy_->State() == State::FREE_BLOCK);
  valid &=
      (right_buddy_ != nullptr) && (right_buddy_->State() == State::FREE_BLOCK);
  PADDLE_ENFORCE(valid == true, "Failed to merge from buddy.");
  state_ = State::FREE_BLOCK;
  raw_ptr_ = left_buddy_;
  size_ = left_buddy_->Size() * 2;
  left_buddy_.reset();
  right_buddy_.reset();
}

void MemoryBlock::UpdateGuards() {
  guard_begin_ = hash(*this, 1);
  guard_end_ = hash(*this, 2);
}

bool MemoryBlock::CheckGuards() const {
  return guard_begin_ == hash(*this, 1) && guard_end_ == hash(*this, 2);
}

}  // namespace detail
}  // namespace memory
}  // namespace paddle
