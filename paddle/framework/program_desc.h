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

#pragma once

#include <memory>
#include <vector>
#include "paddle/framework/block_desc.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/proto_desc.h"
#include "paddle/platform/macros.h"

namespace paddle {
namespace framework {

class BlockDesc;

class ProgramDesc {
 public:
  ProgramDesc();

  explicit ProgramDesc(const proto::ProgramDesc &desc);

  ProgramDesc(const ProgramDesc &o);

  explicit ProgramDesc(const std::string &binary_str);

  BlockDesc *AppendBlock(const BlockDesc &parent);

  BlockDesc *MutableBlock(size_t idx) { return blocks_[idx].get(); }

  const BlockDesc &Block(size_t idx) const { return *blocks_[idx]; }

  size_t Size() const { return blocks_.size(); }

  proto::ProgramDesc *Proto();

  const std::vector<std::string> GetFeedTargetNames();
  const std::vector<std::string> GetFetchTargetNames();

 private:
  proto::ProgramDesc desc_;

  std::vector<std::unique_ptr<BlockDesc>> blocks_;
};
}  // namespace framework
}  // namespace paddle
