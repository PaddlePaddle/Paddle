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

namespace ir {
Block::~Block() { clear(); }
void Block::push_back(Operation *op) {
  op->set_parent(this);
  ops_.push_back(op);
}

void Block::push_front(Operation *op) {
  op->set_parent(this);
  ops_.push_front(op);
}

Block::iterator Block::insert(const_iterator iterator, Operation *op) {
  op->set_parent(this);
  return ops_.insert(iterator, op);
}

void Block::clear() {
  while (!empty()) {
    ops_.back()->destroy();
    ops_.pop_back();
  }
}
}  // namespace ir
