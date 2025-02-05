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

#include "paddle/pir/include/core/region.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/block.h"
#include "paddle/pir/include/core/operation.h"

namespace pir {
Region::~Region() { clear(); }

void Region::push_back(Block *block) { insert(blocks_.end(), block); }

Block &Region::emplace_back() {
  auto block = new Block;
  insert(blocks_.end(), block);
  return *block;
}

void Region::push_front(Block *block) { insert(blocks_.begin(), block); }

Region::Iterator Region::insert(ConstIterator position, Block *block) {
  Region::Iterator iter = blocks_.insert(position, block);
  block->SetParent(this);
  return iter;
}

Region::Iterator Region::erase(ConstIterator position) {
  PADDLE_ENFORCE_EQ(
      position->GetParent(),
      this,
      common::errors::InvalidArgument("iterator not own this region."));
  delete position;
  return blocks_.erase(position);
}

void Region::CloneInto(Region &other, IrMapping &ir_mapping) const {
  if (empty()) {
    return;
  }
  other.clear();
  // clone blocks, block arguments and sub operations
  for (const auto &block : *this) {
    auto new_block = new Block;
    ir_mapping.Add(&block, new_block);
    for (const auto &arg : block.args()) {
      auto new_arg = new_block->AddArg(arg.type());
      ir_mapping.Add(arg, new_arg);
      const BlockArgument &block_arg = arg.dyn_cast<BlockArgument>();
      for (auto &attr : block_arg.attributes()) {
        new_arg.set_attribute(attr.first, attr.second);
      }
    }
    other.push_back(new_block);
  }
  // clone sub operations, but not map operands nor clone regions
  {
    auto clone_options = CloneOptions(false, false, true);
    auto iter = begin();
    auto new_iter = other.begin();
    for (; iter != end(); ++iter, ++new_iter) {
      const Block &block = *iter;
      Block &new_block = *new_iter;
      for (const auto &op : block)
        new_block.push_back(op.Clone(ir_mapping, clone_options));
    }
  }
  // after all operation results are mapped, map operands and clone regions.
  {
    auto iter = begin();
    auto new_iter = other.begin();
    for (; iter != end(); ++iter, ++new_iter) {
      auto op_iter = iter->begin();
      auto new_op_iter = new_iter->begin();
      for (; op_iter != iter->end(); ++op_iter, ++new_op_iter) {
        const Operation &op = *op_iter;
        Operation &new_op = *new_op_iter;
        // operands of new_op are same as op, now map them.
        for (uint32_t i = 0; i < op.num_operands(); ++i)
          new_op.operand(i).set_source(ir_mapping.Lookup(op.operand_source(i)));
        // clone sub regions
        for (uint32_t i = 0; i < op.num_regions(); ++i)
          op.region(i).CloneInto(new_op.region(i), ir_mapping);
      }
    }
  }
}

std::unique_ptr<pir::Block> Region::TakeBack() {
  Block *block = nullptr;
  if (!blocks_.empty()) {
    block = blocks_.back();
    blocks_.pop_back();
  }
  return std::unique_ptr<pir::Block>(block);
}
void Region::TakeBody(Region &&other) {
  clear();
  blocks_.swap(other.blocks_);
  for (auto &block : blocks_) {
    block->SetParent(this);
  }
}

void Region::clear() {
  // In order to ensure the correctness of UD Chain,
  // BlockOperend should be deconstructed before its source.
  for (auto iter = blocks_.rbegin(); iter != blocks_.rend(); ++iter) {
    (*iter)->ClearOps();
  }
  while (!empty()) {
    delete blocks_.back();
    blocks_.pop_back();
  }
}

void Region::swap(Region &&other) {
  blocks_.swap(other.blocks_);
  for (auto &block : *this) {
    block.SetParent(this);
  }
  for (auto &block : other) {
    block.SetParent(&other);
  }
}

template <WalkOrder Order, typename FuncT>
void Region::Walk(FuncT &&callback) {
  for (auto &block : *this) {
    block.Walk<Order>(callback);
  }
}

Program *Region::parent_program() const {
  return parent_ ? parent_->GetParentProgram() : nullptr;
}
IrContext *Region::ir_context() const {
  PADDLE_ENFORCE_NOT_NULL(parent_,
                          common::errors::InvalidArgument(
                              "Region is not attached to a operation."));
  return parent_->ir_context();
}
}  // namespace pir
