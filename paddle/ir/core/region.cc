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

#include "paddle/ir/core/region.h"
#include "paddle/ir/core/block.h"

namespace ir {
Region::~Region() { clear(); }

void Region::push_back(Block *block) { insert(blocks_.end(), block); }

void Region::emplace_back() { push_back(new Block); }

void Region::push_front(Block *block) { insert(blocks_.begin(), block); }

Region::iterator Region::insert(const_iterator position, Block *block) {
  Region::iterator iter = blocks_.insert(position, block);
  block->SetParent(this, iter);
  return iter;
}
void Region::TakeBody(Region &&other) {
  clear();
  blocks_.swap(other.blocks_);
  for (auto iter = blocks_.begin(); iter != blocks_.end(); ++iter) {
    (*iter)->SetParent(this, iter);
  }
}

void Region::clear() {
  while (!empty()) {
    delete blocks_.back();
    blocks_.pop_back();
  }
}
}  // namespace ir
