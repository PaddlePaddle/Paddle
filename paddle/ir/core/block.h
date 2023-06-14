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

#include <cstddef>
#include <list>

#include "paddle/ir/core/region.h"

namespace ir {
class Operation;

class Block {
  using OpListType = std::list<Operation *>;

 public:
  using iterator = OpListType::iterator;
  using reverse_iterator = OpListType::reverse_iterator;
  using const_iterator = OpListType::const_iterator;

  Block() = default;
  ~Block();

  Region *GetParent() const { return parent_; }
  Operation *GetParentOp() const;

  bool empty() const { return ops_.empty(); }
  size_t size() const { return ops_.size(); }

  iterator begin() { return ops_.begin(); }
  iterator end() { return ops_.end(); }
  reverse_iterator rbegin() { return ops_.rbegin(); }
  reverse_iterator rend() { return ops_.rend(); }

  Operation *back() const { return ops_.back(); }
  Operation *front() const { return ops_.front(); }
  void push_back(Operation *op);
  void push_front(Operation *op);
  iterator insert(const_iterator iterator, Operation *op);
  void clear();
  operator Region::iterator() { return position_; }

 private:
  Block(Block &) = delete;
  Block &operator=(const Block &) = delete;

  // Allow access to 'SetParent'.
  friend class Region;
  void SetParent(Region *parent, Region::iterator position);

 private:
  Region *parent_;  // not owned
  OpListType ops_;  // owned
  Region::iterator position_;
};
}  // namespace ir
