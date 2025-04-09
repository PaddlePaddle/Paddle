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

#include "paddle/pir/include/core/block_argument.h"
#include "paddle/pir/include/core/block_operand.h"
#include "paddle/pir/include/core/dll_decl.h"
#include "paddle/pir/include/core/iterator.h"
#include "paddle/pir/include/core/region.h"
#include "paddle/pir/include/core/visitors.h"

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
  const OpListType &ops() const { return ops_; }

  Operation &back() { return *ops_.back(); }
  Operation &front() { return *ops_.front(); }
  const Operation &back() const { return *ops_.back(); }
  const Operation &front() const { return *ops_.front(); }

  void push_back(Operation *op);
  void push_front(Operation *op);
  void pop_back();
  Iterator insert(ConstIterator iterator, Operation *op);
  Iterator erase(ConstIterator position);
  void ClearOps();

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

  // This is a unsafe function, please use it carefully.
  void ResetOpListOrder(const OpListType &new_op_list);

  ///
  /// \brief Position argument management
  ///
  using ArgsType = std::vector<Value>;
  using ArgsIterator = ArgsType::iterator;
  using ConstArgsIterator = ArgsType::const_iterator;

  ArgsIterator args_begin() { return args_.begin(); }
  ArgsIterator args_end() { return args_.end(); }
  ConstArgsIterator args_begin() const { return args_.begin(); }
  ConstArgsIterator args_end() const { return args_.end(); }
  bool args_empty() const { return args_.empty(); }
  uint32_t args_size() const { return args_.size(); }
  const ArgsType &args() const { return args_; }
  Value arg(uint32_t index) const { return args_[index]; }
  Type arg_type(uint32_t index) const { return args_[index].type(); }
  void ClearArgs();
  Value AddArg(Type type);
  void EraseArg(uint32_t index);
  template <class TypeIter>
  void AddArgs(TypeIter first, TypeIter last);
  template <class TypeContainer>
  void AddArgs(const TypeContainer &container) {
    AddArgs(container.begin(), container.end());
  }
  void AddArgs(std::initializer_list<Type> type_list) {
    AddArgs(std::begin(type_list), std::end(type_list));
  }

  ///
  /// \brief Keyword argument management
  ///
  using KwargsType = std::unordered_map<std::string, Value>;
  using KwargsIterator = KwargsType::iterator;
  using ConstKwargsIterator = KwargsType::const_iterator;

  KwargsIterator kwargs_begin() { return kwargs_.begin(); }
  KwargsIterator kwargs_end() { return kwargs_.end(); }
  ConstKwargsIterator kwargs_begin() const { return kwargs_.begin(); }
  ConstKwargsIterator kwargs_end() const { return kwargs_.end(); }
  bool kwargs_empty() const { return kwargs_.empty(); }
  uint32_t kwargs_size() const { return kwargs_.size(); }
  const KwargsType &kwargs() const { return kwargs_; }
  Value kwarg(const std::string &keyword) const { return kwargs_.at(keyword); }
  Type kwarg_type(const std::string &keyword) const {
    return kwarg(keyword).type();
  }
  void ClearKwargs();
  Value AddKwarg(const std::string &keyword, Type type);
  void EraseKwarg(const std::string &keyword);
  bool HasKwarg(const std::string &keyword) const {
    return kwargs_.find(keyword) != kwargs_.end();
  }
  template <class KwTypeIter>
  void AddKwargs(KwTypeIter first, KwTypeIter last);
  template <class KwTypeContainer>
  void AddKwargs(const KwTypeContainer &container) {
    AddKwargs(container.begin(), container.end());
  }

  // Walk the operations in the specified [begin, end) range of this block.
  // PostOrder by default.
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(Block::Iterator begin, Block::Iterator end, FuncT &&callback) {
    for (auto &op = begin; op != end; ++op) {
      pir::Walk<Order>(&*op, callback);
    }
  }

  // Walk the operations in the whole of this block.
  // PostOrder by default.
  template <WalkOrder Order = WalkOrder::PostOrder, typename FuncT>
  void Walk(FuncT &&callback) {
    return Walk<Order>(begin(), end(), std::forward<FuncT>(callback));
  }

  uint32_t num_ops() {
    uint32_t num = 0;
    Walk([&num](Operation *) { ++num; });
    return num;
  }

  OpListType get_recursive_ops() {
    OpListType ops;
    Walk([&ops](Operation *op) { ops.push_back(op); });
    return ops;
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
  OpListType ops_;     // owned
  ArgsType args_;      // owned
  KwargsType kwargs_;  // owned
  Region *parent_;     // not owned
};

std::ostream &operator<<(std::ostream &os, const Block &block);

template <class TypeIter>
void Block::AddArgs(TypeIter first, TypeIter last) {
  while (first != last) {
    AddArg(*first++);
  }
}

template <class KwTypeIter>
void Block::AddKwargs(KwTypeIter first, KwTypeIter last) {
  while (first != last) {
    AddKwarg(first->first, first->second);
    ++first;
  }
}

}  // namespace pir
