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

#include "paddle/pir/core/block_argument.h"
#include "paddle/pir/core/block_operand.h"
#include "paddle/pir/core/dll_decl.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/use_iterator.h"

namespace pir {
class Operation;

class IR_API Block {
  using OpListType = std::list<Operation *>;

 public:
  using Iterator = OpListType::iterator;
  using ReverseIterator = OpListType::reverse_iterator;
  using ConstIterator = OpListType::const_iterator;

  Block() = default;
  ~Block();

  Region *GetParent() const { return parent_; }
  Operation *GetParentOp() const;

  bool empty() const { return ops_.empty(); }
  size_t size() const { return ops_.size(); }

  ConstIterator begin() const { return ops_.begin(); }
  ConstIterator end() const { return ops_.end(); }
  Iterator begin() { return ops_.begin(); }
  Iterator end() { return ops_.end(); }
  ReverseIterator rbegin() { return ops_.rbegin(); }
  ReverseIterator rend() { return ops_.rend(); }

  Operation *back() const { return ops_.back(); }
  Operation *front() const { return ops_.front(); }
  void push_back(Operation *op);
  void push_front(Operation *op);
  Iterator insert(ConstIterator iterator, Operation *op);
  Iterator erase(ConstIterator position);
  void clear();
  operator Region::iterator() { return position_; }

  ///
  /// \brief Provide iterator interface to access Value use chain.
  ///
  using UseIterator = ValueUseIterator<BlockOperand>;
  UseIterator use_begin() const;
  UseIterator use_end() const;
  BlockOperand first_use() const { return first_use_; }
  void set_first_use(BlockOperand first_use) { first_use_ = first_use; }
  bool use_empty() const { return !first_use_; }
  bool HasOneUse() const;
  BlockOperand *first_use_addr() { return &first_use_; }

  // This is a unsafe funcion, please use it carefully.
  void ResetOpListOrder(const OpListType &new_op_list);

  ///
  /// \brief Block argument management
  ///
  using BlockArgListType = std::vector<BlockArgument>;
  using ArgsIterator = BlockArgListType::iterator;

  ArgsIterator args_begin() { return arguments_.begin(); }
  ArgsIterator args_end() { return arguments_.end(); }
  bool args_empty() const { return arguments_.empty(); }
  uint32_t args_size() const { return arguments_.size(); }
  BlockArgument argument(uint32_t index) { return arguments_[index]; }
  Type argument_type(uint32_t index) const { return arguments_[index].type(); }

  void ClearArguments();
  void AddArgument(Type type);
  template <class TypeIter>
  void AddArguments(TypeIter first, TypeIter last);

  template <class TypeContainer>
  void AddArguments(const TypeContainer &container) {
    AddArguments(container.begin(), container.end());
  }

 private:
  Block(Block &) = delete;
  Block &operator=(const Block &) = delete;

  // Allow access to 'SetParent'.
  friend class Region;
  void SetParent(Region *parent, Region::iterator position);

  static bool TopoOrderCheck(const OpListType &op_list);

 private:
  Region::iterator position_;
  BlockOperand first_use_;
  OpListType ops_;              // owned
  BlockArgListType arguments_;  // owned
  Region *parent_;              // not owned
};

template <class TypeIter>
void Block::AddArguments(TypeIter first, TypeIter last) {
  while (first != last) {
    AddArgument(*first++);
  }
}

}  // namespace pir
