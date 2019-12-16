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
#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <vector>

namespace paddle {
namespace memory {
namespace detail {

class MemoryBlockPool;

// MemoryBlock represents Each allocated memory block, which contains
// MemoryBlock::Desc and the payload.
class MemoryBlock {
 public:
  friend struct MemoryBlockComparator;

 public:
  enum Type {
    FREE_CHUNK,    // memory is free and idle
    ARENA_CHUNK,   // memory is being occupied
    HUGE_CHUNK,    // memory is out of management
    INVALID_CHUNK  // memory is invalid
  };

  MemoryBlock() {}

  explicit MemoryBlock(size_t size)
      : data_(nullptr),
        type_(Type::INVALID_CHUNK),
        index_(-1),
        size_(size),
        left_buddy_(nullptr),
        right_buddy_(nullptr) {}

  MemoryBlock(void* data, Type t, size_t index, size_t size,
              MemoryBlock* left_buddy, MemoryBlock* right_buddy)
      : data_(data),
        type_(t),
        index_(index),
        size_(size),
        left_buddy_(left_buddy),
        right_buddy_(right_buddy) {}

  void Init(void* data, Type type, size_t index, size_t size,
            MemoryBlock* left_buddy, MemoryBlock* right_buddy);

  // Split the allocation into left/right blocks.
  // return false if the request size equal to the block size
  bool Split(size_t size, MemoryBlockPool* pool);

  // Merge left and right blocks together.
  void Merge(MemoryBlock* right_buddy, MemoryBlockPool* pool);

  // Mark the allocation as free.
  inline void MarkAsFree() {
    this->type_ = FREE_CHUNK;
    this->UpdateGuards();
  }

  // mutator for type
  inline void set_type(const MemoryBlock::Type& type) {
    this->type_ = type;
    this->UpdateGuards();
  }

  // accessor for data_
  void* get_data() const { return this->data_; }

  // accessor for type_
  inline const MemoryBlock::Type& get_type() const { return this->type_; }

  // accessor for index_
  inline const size_t& get_index() const { return this->index_; }

  // accessor for size_
  inline const size_t& get_size() const { return this->size_; }

  // accessor for left_buddy_
  inline MemoryBlock* get_left_buddy() const { return this->left_buddy_; }

  // accessor for right_buddy_
  inline MemoryBlock* get_right_buddy() const { return this->right_buddy_; }

  // Updates guard_begin and guard_end by hashes of the Metadata object.
  void UpdateGuards();

  // Checks that guard_begin and guard_end are hashes of the Metadata object.
  bool CheckGuards() const;

 private:
  size_t guard_begin = 0;
  void* data_ = nullptr;
  MemoryBlock::Type type_ = MemoryBlock::INVALID_CHUNK;
  size_t index_ = 0;
  size_t size_ = 0;
  MemoryBlock* left_buddy_ = nullptr;
  MemoryBlock* right_buddy_ = nullptr;
  size_t guard_end = 0;
};

class MemoryBlockPool {
 public:
  explicit MemoryBlockPool(const int inc_size = 100) : inc_size_(inc_size) {}
  ~MemoryBlockPool();

 public:
  // pop a free MemoryBlock pointer from queue
  MemoryBlock* Get();

  // return the pointer back to queue
  void Push(MemoryBlock*);

 private:
  int inc_size_;
  std::queue<MemoryBlock*> pool_;
  std::vector<MemoryBlock*> raw_pointers_;
};

struct MemoryBlockComparator {
  bool operator()(const MemoryBlock* a, const MemoryBlock* b) const {
    return (a->size_ == b->size_) ? a->data_ < b->data_ : a->size_ < b->size_;
  }
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
