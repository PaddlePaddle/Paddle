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

#include "paddle/ir/core/block.h"

#include <unordered_set>

#include "paddle/ir/core/enforce.h"
#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/region.h"

namespace ir {
Block::~Block() {
  assert(use_empty() && "block destroyed still has uses.");
  clear();
}
void Block::push_back(Operation *op) { insert(ops_.end(), op); }

void Block::push_front(Operation *op) { insert(ops_.begin(), op); }

Operation *Block::GetParentOp() const {
  return parent_ ? parent_->GetParent() : nullptr;
}

Block::iterator Block::insert(const_iterator iterator, Operation *op) {
  Block::iterator iter = ops_.insert(iterator, op);
  op->SetParent(this, iter);
  return iter;
}

Block::iterator Block::erase(const_iterator position) {
  IR_ENFORCE((*position)->GetParent() == this, "iterator not own this block.");
  (*position)->Destroy();
  return ops_.erase(position);
}

void Block::clear() {
  while (!empty()) {
    ops_.back()->Destroy();
    ops_.pop_back();
  }
}

void Block::SetParent(Region *parent, Region::iterator position) {
  parent_ = parent;
  position_ = position;
}

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

bool Block::TopoOrderCheck(const OpListType &op_list) {
  std::unordered_set<Value> visited_values;
  for (const Operation *op : op_list) {
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

}  // namespace ir
