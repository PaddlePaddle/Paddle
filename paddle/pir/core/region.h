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
#include <memory>

#include "paddle/pir/core/dll_decl.h"
#include "paddle/pir/core/iterator.h"
#include "paddle/pir/core/visitors.h"

namespace pir {

class Block;
class Operation;
class IrContext;
class Program;

class IR_API Region {
 public:
  using Element = Block;
  using Iterator = PointerListIterator<Block>;
  using ConstIterator = PointerListConstIterator<Block>;
  using ReverseIterator = std::reverse_iterator<Iterator>;
  using ConstReverseIterator = std::reverse_iterator<ConstIterator>;

  explicit Region(Operation *op = nullptr) : parent_(op) {}
  Region(const Region &) = delete;
  Region &operator=(const Region &) = delete;
  ~Region();
  bool empty() const { return blocks_.empty(); }
  size_t size() const { return blocks_.size(); }

  Iterator begin() { return blocks_.begin(); }
  Iterator end() { return blocks_.end(); }
  ConstIterator begin() const { return blocks_.begin(); }
  ConstIterator end() const { return blocks_.end(); }
  ReverseIterator rbegin() { return blocks_.rbegin(); }
  ReverseIterator rend() { return blocks_.rend(); }
  ConstReverseIterator rbegin() const { return blocks_.rbegin(); }
  ConstReverseIterator rend() const { return blocks_.rend(); }

  Block &front() { return *blocks_.front(); }
  Block &back() { return *blocks_.back(); }

  const Block &front() const { return *blocks_.front(); }
  const Block &back() const { return *blocks_.back(); }

  void push_back(Block *block);
  Block &emplace_back();
  void push_front(Block *block);
  Iterator insert(ConstIterator position, Block *block);
  Iterator erase(ConstIterator position);
  void clear();

  /// Operation Walkers, walk the operations in this region. The callback method
  /// is called for each nested region, block or operation,
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(FuncT &&callback) {
    for (auto &block : *this) {
      // block.Walk<Order>(callback);
      auto begin = block.begin();
      auto end = block.end();
      for (auto &op = begin; op != end; ++op) {
        detail::Walk<Order>(&*op, callback);
      }
    }
  }

  // take the last block of region.
  // if region is empty, return nullptr;
  std::unique_ptr<Block> TakeBack();
  void TakeBody(Region &&other);

  Operation *GetParent() const { return parent_; }
  void set_parent(Operation *parent) { parent_ = parent; }
  // return the program which contains this region.
  // if region is not in a program, return nullptr.
  Program *parent_program() const;

  IrContext *ir_context() const;

 private:
  Operation *parent_{nullptr};  // not owned
  std::list<Block *> blocks_;   // owned
};
}  // namespace pir
