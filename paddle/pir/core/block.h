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
#include "paddle/pir/core/iterator.h"
#include "paddle/pir/core/region.h"
#include "paddle/pir/core/visitors.h"

namespace pir {
class Operation;
class Program;

class IR_API Block {
  using OpListType = std::list<Operation *>;

 public:
  using Iterator = PointerListIterator<Operation>;
  using ConstIterator = PointerListConstIterator<Operation>;

  using ReverseIterator = std::reverse_iterator<Iterator>;
  using ConstReverseIterator = std::reverse_iterator<ConstIterator>;

  Block() = default;
  ~Block();

  Region *GetParent() const { return parent_; }
  Operation *GetParentOp() const;

  // return the program which contains this block.
  // if block is not in a program, return nullptr.
  Program *parent_program() const {
    return parent_ ? parent_->parent_program() : nullptr;
  }

  bool empty() const { return ops_.empty(); }
  size_t size() const { return ops_.size(); }

  ConstIterator begin() const { return ops_.begin(); }
  ConstIterator end() const { return ops_.end(); }
  Iterator begin() { return ops_.begin(); }
  Iterator end() { return ops_.end(); }
  ConstReverseIterator rbegin() const { return ops_.rbegin(); }
  ConstReverseIterator rend() const { return ops_.rend(); }
  ReverseIterator rbegin() { return ops_.rbegin(); }
  ReverseIterator rend() { return ops_.rend(); }

  Operation &back() { return *ops_.back(); }
  Operation &front() { return *ops_.front(); }
  const Operation &back() const { return *ops_.back(); }
  const Operation &front() const { return *ops_.front(); }

  void push_back(Operation *op);
  void push_front(Operation *op);
  void pop_back();
  Iterator insert(ConstIterator iterator, Operation *op);
  Iterator erase(ConstIterator position);
  void clear();

  // Assign the operation underlying in position with parameter op,
  // meanwhile, destroy the original operation.
  void Assign(Iterator position, Operation *op);

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
  using ArgListType = std::vector<Value>;
  using ArgsIterator = ArgListType::iterator;
  using ConstArgsIterator = ArgListType::const_iterator;

  ArgsIterator args_begin() { return arguments_.begin(); }
  ArgsIterator args_end() { return arguments_.end(); }
  ConstArgsIterator args_begin() const { return arguments_.begin(); }
  ConstArgsIterator args_end() const { return arguments_.end(); }
  bool args_empty() const { return arguments_.empty(); }
  uint32_t args_size() const { return arguments_.size(); }
  const ArgListType &args() const { return arguments_; }
  Value arg(uint32_t index) { return arguments_[index]; }
  Type arg_type(uint32_t index) const { return arguments_[index].type(); }
  void ClearArguments();
  Value AddArgument(Type type);
  void EraseArgument(uint32_t index);
  template <class TypeIter>
  void AddArguments(TypeIter first, TypeIter last);
  template <class TypeContainer>
  void AddArguments(const TypeContainer &container) {
    AddArguments(container.begin(), container.end());
  }
  void AddArguments(std::initializer_list<Type> type_list) {
    AddArguments(std::begin(type_list), std::end(type_list));
  }

  // Walk the operations in the specified [begin, end) range of this block.
  // PostOrder by default.
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(Block::Iterator begin, Block::Iterator end, FuncT &&callback) {
    for (auto &op = begin; op != end; ++op) {
      detail::Walk<Order>(&*op, callback);
    }
  }

  // Walk the operations in the whole of this block.
  // PostOrder by default.
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(FuncT &&callback) {
    return Walk<Order>(begin(), end(), std::forward<FuncT>(callback));
  }

 private:
  Block(Block &) = delete;
  Block &operator=(const Block &) = delete;

  // Allow access to 'SetParent'.
  friend class Region;
  void SetParent(Region *parent);

  // Take out corresponding Operation and its ownership.
  friend class Operation;
  Operation *Take(Operation *op);

  static bool TopoOrderCheck(const OpListType &op_list);

 private:
  BlockOperand first_use_;
  OpListType ops_;         // owned
  ArgListType arguments_;  // owned
  Region *parent_;         // not owned
};

template <class TypeIter>
void Block::AddArguments(TypeIter first, TypeIter last) {
  while (first != last) {
    AddArgument(*first++);
  }
}

}  // namespace pir
