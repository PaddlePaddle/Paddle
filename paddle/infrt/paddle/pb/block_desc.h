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

#include "paddle/infrt/paddle/cpp/desc_api.h"
#include "paddle/infrt/paddle/framework.pb.h"

namespace infrt::paddle::pb {

namespace framework_proto = ::paddle::framework::proto;

class BlockDesc : public cpp::BlockDescAPI {
 public:
  BlockDesc() = delete;

  explicit BlockDesc(framework_proto::BlockDesc* desc) : desc_(desc) {
    CHECK(desc_);
  }

  framework_proto::BlockDesc* Proto() { return desc_; }

  const framework_proto::BlockDesc& ReadonlyProto() const { return *desc_; }

  int32_t Idx() const override { return desc_->idx(); }

  void SetIdx(int32_t idx) override { desc_->set_idx(idx); }

  int32_t ParentIdx() const override { return desc_->parent_idx(); }

  void SetParentIdx(int32_t idx) override { desc_->set_parent_idx(idx); }

  size_t VarsSize() const override { return desc_->vars_size(); }

  void ClearVars() override { desc_->clear_vars(); }

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T* AddVar();

  size_t OpsSize() const override { return desc_->ops_size(); }

  void ClearOps() override { desc_->clear_ops(); }

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T* AddOp();

  int32_t ForwardBlockIdx() const override {
    return desc_->forward_block_idx();
  }

  void SetForwardBlockIdx(int32_t idx) override {
    desc_->set_forward_block_idx(idx);
  }

 private:
  framework_proto::BlockDesc* desc_;  // not_own
};

}  // namespace infrt::paddle::pb
