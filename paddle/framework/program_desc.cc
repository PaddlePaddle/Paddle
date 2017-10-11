/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/program_desc.h"
#include "paddle/framework/block_desc.h"

namespace paddle {
namespace framework {

using ProgDescMap =
    std::unordered_map<ProgramDesc *, std::unique_ptr<ProgramDescBind>>;
static ProgDescMap *g_bind_map = nullptr;

ProgramDescBind &ProgramDescBind::Instance(ProgramDesc *prog) {
  if (g_bind_map == nullptr) {
    g_bind_map = new ProgDescMap();
  }
  auto &map = *g_bind_map;
  auto &ptr = map[prog];

  if (ptr == nullptr) {
    ptr.reset(new ProgramDescBind(prog));
  }
  return *ptr;
}

BlockDescBind *ProgramDescBind::AppendBlock(const BlockDescBind &parent) {
  auto *b = prog_->add_blocks();
  b->set_parent_idx(parent.ID());
  b->set_idx(prog_->blocks_size() - 1);
  blocks_.emplace_back(new BlockDescBind(this, b));
  return blocks_.back().get();
}

ProgramDesc *ProgramDescBind::Proto() {
  for (auto &block : blocks_) {
    block->Sync();
  }
  return prog_;
}

ProgramDescBind::ProgramDescBind(ProgramDesc *prog) {
  prog_ = prog;
  for (auto &block : *prog->mutable_blocks()) {
    blocks_.emplace_back(new BlockDescBind(this, &block));
  }
}
}  // namespace framework
}  // namespace paddle
