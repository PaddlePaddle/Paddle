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

BlockDescBind *ProgramDescBind::AppendBlock(const BlockDescBind &parent) {
  auto *b = desc_.add_blocks();
  b->set_parent_idx(parent.ID());
  b->set_idx(desc_.blocks_size() - 1);
  blocks_.emplace_back(new BlockDescBind(this, b));
  return blocks_.back().get();
}

ProgramDesc *ProgramDescBind::Proto() {
  for (auto &block : blocks_) {
    block->Flush();
  }
  return &desc_;
}

ProgramDescBind::ProgramDescBind() {
  auto *block = desc_.mutable_blocks()->Add();
  block->set_idx(kRootBlockIndex);
  block->set_parent_idx(kNoneBlockIndex);
  blocks_.emplace_back(new BlockDescBind(this, block));
}

ProgramDescBind::ProgramDescBind(const ProgramDescBind &o) {
  desc_ = o.desc_;

  for (int i = 0; i < desc_.blocks_size(); ++i) {
    auto *block = desc_.mutable_blocks(i);
    blocks_.emplace_back(new BlockDescBind(*o.blocks_[i], block, this));
  }
}

ProgramDescBind::ProgramDescBind(const ProgramDesc &desc) {
  desc_ = desc;
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDescBind(this, &block_desc));
  }
}

ProgramDescBind::ProgramDescBind(const std::string &binary_str) {
  PADDLE_ENFORCE(desc_.ParseFromString(binary_str),
                 "Fail to parse program_desc from binary string.");
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDescBind(this, &block_desc));
  }
}

}  // namespace framework
}  // namespace paddle
