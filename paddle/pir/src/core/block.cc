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

#include "paddle/pir/include/core/block.h"

#include <glog/logging.h>
#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/operation.h"
#include "paddle/pir/include/core/region.h"

namespace pir {
Block::~Block() {  // NOLINT
  if (!use_empty()) {
    auto parent_op = GetParentOp();
    PADDLE_FATAL(
        "Destroyed a block that is still in use.. The parent op is : %s",
        parent_op ? parent_op->name() : std::string("nullptr"));
  }
  ClearOps();
  ClearKwargs();
  ClearArgs();
}
void Block::push_back(Operation *op) { insert(ops_.end(), op); }

void Block::push_front(Operation *op) { insert(ops_.begin(), op); }

void Block::pop_back() {
  PADDLE_ENFORCE_EQ(
      !ops_.empty(),
      true,
      common::errors::InvalidArgument("can't pop back from empty block."));
  ops_.back()->Destroy();
  ops_.pop_back();
}

Operation *Block::GetParentOp() const {
  return parent_ ? parent_->GetParent() : nullptr;
}

Block::Iterator Block::insert(ConstIterator iterator, Operation *op) {
  Block::Iterator iter = ops_.insert(iterator, op);
  op->SetParent(this, iter);
  return iter;
}

Block::Iterator Block::erase(ConstIterator position) {
  PADDLE_ENFORCE_EQ(
      position->GetParent(),
      this,
      common::errors::InvalidArgument("iterator not own this block."));
  position->Destroy();
  return ops_.erase(position);
}

void Block::ClearOps() {
  while (!empty()) {
    pop_back();
  }
}

void Block::Assign(Iterator position, Operation *op) {
  PADDLE_ENFORCE_EQ(
      position->GetParent(),
      this,
      common::errors::InvalidArgument("position not own this block."));
  position->Destroy();
  position.set_underlying_pointer(op);
  op->SetParent(this, position);
}

Operation *Block::Take(Operation *op) {
  PADDLE_ENFORCE_EQ(
      op && op->GetParent() == this,
      true,
      common::errors::InvalidArgument("iterator not own this block."));
  ops_.erase(Iterator(*op));
  return op;
}

void Block::SetParent(Region *parent) { parent_ = parent; }

Block::UseIterator Block::use_begin() const { return first_use_; }

Block::UseIterator Block::use_end() const { return Block::UseIterator(); }

bool Block::HasOneUse() const { return first_use_ && !first_use_.next_use(); }

void Block::ResetOpListOrder(const OpListType &new_op_list) {
  PADDLE_ENFORCE_EQ(new_op_list.size(),
                    ops_.size(),
                    common::errors::InvalidArgument(
                        "The size of new_op_list not same with ops_."));
  PADDLE_ENFORCE_EQ(TopoOrderCheck(new_op_list),
                    true,
                    common::errors::InvalidArgument(
                        "The new_op_list is not in topological order."));

  ops_.clear();
  for (Operation *op : new_op_list) {
    push_back(op);
  }
}

void Block::ClearArgs() {
  for (auto &arg : args_) {
    arg.dyn_cast<BlockArgument>().Destroy();
  }
  args_.clear();
}

Value Block::AddArg(Type type) {
  auto argument = BlockArgument::Create(type, this, args_.size());
  args_.emplace_back(argument);
  return argument;
}

void Block::EraseArg(uint32_t index) {
  auto argument = arg(index);
  PADDLE_ENFORCE_EQ(argument.use_empty(),
                    true,
                    common::errors::InvalidArgument(
                        "Erase a block argument that is still in use."));
  argument.dyn_cast<BlockArgument>().Destroy();
  args_.erase(args_.begin() + index);
}

void Block::ClearKwargs() {
  for (auto &kwarg : kwargs_) {
    kwarg.second.dyn_cast<BlockArgument>().Destroy();
  }
  kwargs_.clear();
}

Value Block::AddKwarg(const std::string &keyword, Type type) {
  PADDLE_ENFORCE_EQ(kwargs_.find(keyword),
                    kwargs_.end(),
                    common::errors::InvalidArgument(
                        "Add keyword (%s) argument which has been existed.",
                        keyword.c_str()));
  auto arg = BlockArgument::Create(type, this, keyword);
  kwargs_[keyword] = arg;
  return arg;
}

void Block::EraseKwarg(const std::string &keyword) {
  PADDLE_ENFORCE_NE(kwargs_.find(keyword),
                    kwargs_.end(),
                    common::errors::InvalidArgument(
                        "Erase keyword (%s) argument which doesn't existed.",
                        keyword.c_str()));
  auto kwarg = kwargs_[keyword];
  PADDLE_ENFORCE_EQ(
      kwarg.use_empty(),
      true,
      common::errors::InvalidArgument(
          "Erase a block keyword argument that is still in use."));
  kwarg.dyn_cast<BlockArgument>().Destroy();
  kwargs_.erase(keyword);
}
bool Block::TopoOrderCheck(const OpListType &op_list) {
  std::unordered_set<Value> visited_values;
  for (Operation *op : op_list) {
    if (op->num_operands() > 0) {
      for (size_t i = 0; i < op->num_operands(); ++i) {
        auto operand = op->operand_source(i);
        if (operand && visited_values.count(op->operand_source(i)) == 0) {
          return false;
        }
      }
    }
    for (size_t i = 0; i < op->results().size(); ++i) {
      visited_values.insert(op->result(i));
    }
  }
  return true;
}

}  // namespace pir
