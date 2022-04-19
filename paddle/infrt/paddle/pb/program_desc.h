// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <glog/logging.h>

#include <string>
#include <vector>

#include "paddle/infrt/paddle/cpp/desc_api.h"
#include "paddle/infrt/paddle/framework.pb.h"

namespace infrt {
namespace paddle {
namespace pb {
namespace framework_proto = ::paddle::framework::proto;

class ProgramDesc : public cpp::ProgramDescAPI {
 public:
  ProgramDesc() = delete;

  explicit ProgramDesc(framework_proto::ProgramDesc *desc) : desc_(desc) {
    CHECK(desc_);
  }

  framework_proto::ProgramDesc *Proto() { return desc_; }

  const framework_proto::ProgramDesc &ReadonlyProto() const { return *desc_; }

  size_t BlocksSize() const override { return desc_->blocks_size(); }

  void ClearBlocks() override { desc_->clear_blocks(); }

  template <typename T>
  T *GetBlock(int32_t idx);

  template <typename T>
  T *AddBlock();

  bool HasVersion() const override { return desc_->has_version(); }

  int64_t Version() const override { return desc_->version().version(); }

  void SetVersion(int64_t version) override {
    desc_->mutable_version()->set_version(version);
  }

 private:
  framework_proto::ProgramDesc *desc_;  // not_own
};

}  // namespace pb
}  // namespace paddle
}  // namespace infrt
