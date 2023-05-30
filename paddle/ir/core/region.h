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

namespace ir {

class Block;
class Operation;

class Region {
 public:
  using iterator = std::list<Block *>::iterator;
  using reverse_iterator = std::list<Block *>::reverse_iterator;
  using const_iterator = std::list<Block *>::const_iterator;
  ~Region();
  Region() = default;

  bool empty() const { return blocks_.empty(); }
  size_t size() const { return blocks_.size(); }

  iterator begin() { return blocks_.begin(); }
  iterator end() { return blocks_.end(); }
  reverse_iterator rbegin() { return blocks_.rbegin(); }
  reverse_iterator rend() { return blocks_.rend(); }

  Block *back() const { return blocks_.back(); }
  Block *front() const { return blocks_.front(); }
  void push_back(Block *block);
  void push_front(Block *block);
  iterator insert(const_iterator position, Block *block);
  void clear();

  void TakeBody(Region &&other);

 private:
  Region(Region &) = delete;
  Region &operator=(const Region &) = delete;
  friend class Operation;
  explicit Region(Operation *op) : parent_(op) {}

 private:
  Operation *parent_{nullptr};  // not owned
  std::list<Block *> blocks_;   // owned
};
}  // namespace ir
