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

#pragma once

#include <memory>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace memory {
namespace detail {

class MemoryBlock {
 public:
  enum State {
    FREE_BLOCK,     // idle memory block
    USED_BLOCK,     // in used memory block
    INVALID_BLOCK,  // not own memory or memory can not be used.
  };
  MemoryBlock();
  MemoryBlock(void* block, uint64_t size);
  ~MemoryBlock();

  void SplitToBuddy();
  void MergeFromBuddy();

  State GetState() const { return state_; }
  void SetState(State state) { state_ = state; }
  MemoryBlock* LeftBuddy() const { return left_buddy_; }
  MemoryBlock* RightBuddy() const { return right_buddy_; }
  uint64_t Size() const { return size_; }

 private:
  void UpdateGuards();
  bool CheckGuards() const;
  void Reset();

 private:
  // guard begin/end is magic number,
  // check the memory when alloc/release
  uint64_t guard_begin_;
  State state_;
  void* raw_ptr_;  // do not own
  // std::unique_ptr<void> raw_ptr_;
  uint64_t size_;
  std::unique_ptr<MemoryBlock> left_buddy_;
  std::unique_ptr<MemoryBlock> right_buddy_;
  uint64_t guard_end_;
  DISABLE_COPY_AND_ASSIGN(MemoryBlock);
};

}  // namespace detail
}  // namespace memory
}  // namespace paddle
