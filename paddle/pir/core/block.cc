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

#include "paddle/pir/core/block.h"

#include <unordered_set>

#include "paddle/common/enforce.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/region.h"

namespace pir {
Block::~Block() {
  if (!use_empty()) {
    LOG(FATAL) << "Destroyed a block that is still in use.";
  }
  ClearOps();
  ClearKwargs();
  ClearArgs();
}
void Block::push_back(Operation *op) { insert(ops_.end(), op); }

void Block::push_front(Operation *op) { insert(ops_.begin(), op); }

void Block::pop_back() {
  IR_ENFORCE(!ops_.empty(), "can't pop back from empty block.");
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
  IR_ENFORCE(position->GetParent() == this, "iterator not own this block.");
  position->Destroy();
  return ops_.erase(position);
}

void Block::ClearOps() {
  while (!empty()) {
    pop_back();
  }
}

void Block::Assign(Iterator position, Operation *op) {
  IR_ENFORCE(position->GetParent() == this, "position not own this block.");
  position->Destroy();
  position.set_underlying_pointer(op);
  op->SetParent(this, position);
}

Operation *Block::Take(Operation *op) {
  IR_ENFORCE(op && op->GetParent() == this, "iterator not own this block.");
  ops_.erase(Iterator(*op));
  return op;
}

void Block::SetParent(Region *parent) { parent_ = parent; }

Block::UseIterator Block::use_begin() const { return first_use_; }

Block::UseIterator Block::use_end() const { return Block::UseIterator(); }

bool Block::HasOneUse() const { return first_use_ && !first_use_.next_use(); }

void Block::ResetOpListOrder(const OpListType &new_op_list) {
  IR_ENFORCE(new_op_list.size() == ops_.size(),
             "The size of new_op_list not same with ops_.");
  IR_ENFORCE(TopoOrderCheck(new_op_list),
             "The new_op_list is not in topological order.");

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
  IR_ENFORCE(argument.use_empty(),
             "Erase a block argument that is still in use.");
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
  IR_ENFORCE(kwargs_.find(keyword) == kwargs_.end(),
             "Add keyword (%s) argument which has been existed.",
             keyword.c_str());
  auto arg = BlockArgument::Create(type, this, 0);
  kwargs_[keyword] = arg;
  return arg;
}

void Block::EraseKwarg(const std::string &keyword) {
  IR_ENFORCE(kwargs_.find(keyword) != kwargs_.end(),
             "Erase keyword (%s) argument which doesn't existed.",
             keyword.c_str());
  auto kwarg = kwargs_[keyword];
  IR_ENFORCE(kwarg.use_empty(),
             "Erase a block keyword argument that is still in use.");
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
