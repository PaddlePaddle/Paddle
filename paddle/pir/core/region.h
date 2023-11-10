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

#include "paddle/pir/core/dll_decl.h"

namespace pir {

class Block;
class Operation;
class IrContext;

class IR_API Region {
 public:
  using iterator = std::list<Block *>::iterator;
  using reverse_iterator = std::list<Block *>::reverse_iterator;
  using const_iterator = std::list<Block *>::const_iterator;
  explicit Region(Operation *op = nullptr) : parent_(op) {}
  Region(const Region &) = delete;
  Region &operator=(const Region &) = delete;
  ~Region();
  bool empty() const { return blocks_.empty(); }
  size_t size() const { return blocks_.size(); }

  iterator begin() { return blocks_.begin(); }
  iterator end() { return blocks_.end(); }
  const_iterator begin() const { return blocks_.begin(); }
  const_iterator end() const { return blocks_.end(); }
  reverse_iterator rbegin() { return blocks_.rbegin(); }
  reverse_iterator rend() { return blocks_.rend(); }

  Block *back() const { return blocks_.back(); }
  Block *front() const { return blocks_.front(); }
  void push_back(Block *block);
  Block *emplace_back();
  void push_front(Block *block);
  iterator insert(const_iterator position, Block *block);
  iterator erase(const_iterator position);
  void clear();

  void TakeBody(Region &&other);

  Operation *GetParent() const { return parent_; }
  void set_parent(Operation *parent) { parent_ = parent; }

  IrContext *ir_context() const;

 private:
  Operation *parent_{nullptr};  // not owned
  std::list<Block *> blocks_;   // owned
};
}  // namespace pir
