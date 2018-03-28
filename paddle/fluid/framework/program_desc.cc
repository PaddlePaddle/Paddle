/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"

namespace paddle {
namespace framework {

BlockDesc *ProgramDesc::AppendBlock(const BlockDesc &parent) {
  auto *b = desc_.add_blocks();
  b->set_parent_idx(parent.ID());
  b->set_idx(desc_.blocks_size() - 1);
  blocks_.emplace_back(new BlockDesc(this, b));
  return blocks_.back().get();
}

proto::ProgramDesc *ProgramDesc::Proto() {
  for (auto &block : blocks_) {
    block->Flush();
  }
  return &desc_;
}

ProgramDesc::ProgramDesc() {
  auto *block = desc_.mutable_blocks()->Add();
  block->set_idx(kRootBlockIndex);
  block->set_parent_idx(kNoneBlockIndex);
  blocks_.emplace_back(new BlockDesc(this, block));
}

ProgramDesc::ProgramDesc(const ProgramDesc &o) {
  desc_ = o.desc_;
  for (int i = 0; i < desc_.blocks_size(); ++i) {
    auto *block = desc_.mutable_blocks(i);
    blocks_.emplace_back(new BlockDesc(*o.blocks_[i], block, this));
  }
  for (auto &block : blocks_) {
    for (auto *op : block->AllOps()) {
      for (const auto &attr : op->Proto()->attrs()) {
        if (attr.type() == proto::AttrType::BLOCK) {
          size_t blk_idx = attr.block_idx();
          op->SetBlockAttr(attr.name(), *this->MutableBlock(blk_idx));
        }
      }
    }
  }
}

ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc) {
  desc_ = desc;
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDesc(this, &block_desc));
  }
  for (auto &block : blocks_) {
    for (auto *op : block->AllOps()) {
      for (const auto &attr : op->Proto()->attrs()) {
        if (attr.type() == proto::AttrType::BLOCK) {
          size_t blk_idx = attr.block_idx();
          op->SetBlockAttr(attr.name(), *this->MutableBlock(blk_idx));
        }
      }
    }
  }
}

ProgramDesc::ProgramDesc(const std::string &binary_str) {
  PADDLE_ENFORCE(desc_.ParseFromString(binary_str),
                 "Fail to parse program_desc from binary string.");
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDesc(this, &block_desc));
  }
}

const std::vector<std::string> ProgramDesc::GetFeedTargetNames() {
  BlockDesc *global_block = blocks_[0].get();
  std::vector<std::string> feed_target_names;
  for (auto *op : global_block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      feed_target_names.insert(feed_target_names.begin(), op->Output("Out")[0]);
    }
  }
  return feed_target_names;
}

const std::vector<std::string> ProgramDesc::GetFetchTargetNames() {
  BlockDesc *global_block = blocks_[0].get();
  std::vector<std::string> fetch_target_names;
  for (auto *op : global_block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      fetch_target_names.push_back(op->Input("X")[0]);
    }
  }
  return fetch_target_names;
}

}  // namespace framework
}  // namespace paddle
