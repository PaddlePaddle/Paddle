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

#include "paddle/pir/core/region.h"
#include "paddle/common/enforce.h"
#include "paddle/pir/core/block.h"
#include "paddle/pir/core/operation.h"

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
  block->SetParent(this, iter);
  return iter;
}

Region::Iterator Region::erase(ConstIterator position) {
  IR_ENFORCE(position->GetParent() == this, "iterator not own this region.");
  delete position;
  return blocks_.erase(position);
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
  for (auto iter = blocks_.begin(); iter != blocks_.end(); ++iter) {
    (*iter)->SetParent(this, iter);
  }
}

void Region::clear() {
  // In order to ensure the correctness of UD Chain,
  // BlockOperend should be decontructed bofore its source.
  for (auto iter = blocks_.rbegin(); iter != blocks_.rend(); ++iter) {
    (*iter)->clear();
  }
  while (!empty()) {
    delete blocks_.back();
    blocks_.pop_back();
  }
}
Program *Region::parent_program() const {
  return parent_ ? parent_->GetParentProgram() : nullptr;
}
IrContext *Region::ir_context() const {
  IR_ENFORCE(parent_, "Region is not attached to a operation.");
  return parent_->ir_context();
}
}  // namespace pir
