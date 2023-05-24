// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include "paddle/ir/operation.h"

namespace ir {
class Block {
 public:
  using iterator = std::list<Operation *>::iterator;
  using reverse_iterator = std::list<Operation *>::reverse_iterator;

  Block() = default;
  ~Block();

  std::list<Operation *> &operations() { return ops_; }

  bool empty() const { return ops_.empty(); }
  size_t size() const { return ops_.size(); }

  iterator begin() { return ops_.begin(); }
  iterator end() { return ops_.end(); }
  reverse_iterator rbegin() { return ops_.rbegin(); }
  reverse_iterator rend() { return ops_.rend(); }

  Operation *back() { return ops_.back(); }
  Operation *front() { return ops_.front(); }
  void push_back(Operation *op) { ops_.push_back(op); }
  void push_front(Operation *op) { ops_.push_front(op); }
  void clear();

 private:
  Block(Block &) = delete;
  void operator=(Block &) = delete;

 private:
  std::list<Operation *> ops_;  // owned
};
}  // namespace ir
