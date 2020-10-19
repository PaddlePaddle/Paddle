// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_functor.h"

namespace paddle {
namespace framework {
class Scope;
namespace ir {
class MemOptVarInfo;
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

class ComputationOpHandle;

class ShareTensorBufferOpHandle : public OpHandleBase {
 public:
  ShareTensorBufferOpHandle(
      ir::Node *node, Scope *scope, size_t scope_idx,
      const std::string &op_type,
      const std::vector<const ir::MemOptVarInfo *> &in_vars_infos,
      const std::vector<std::string> &out_var_names, bool share_dims = false);

  std::unordered_map<std::string, std::string> ReusedVars() const;

  Priority GetPriority() const override { return Priority::kHighest; }

  size_t GetScopeIdx() const { return functor_.GetScopeIdx(); }

  void AddReuseVarPair(const ir::MemOptVarInfo *in_var_info,
                       const std::string &out_var_name);

  void SetShareDims(bool share_dims);

  const ShareTensorBufferFunctor &Functor() const { return functor_; }

 protected:
  std::string Name() const override { return "buffer_share"; }

  void RunImpl() final;

  void InitCUDA() override;

  std::vector<Scope *> GetLocalScopes() override {
    return {functor_.GetScope()};
  }

 private:
  ShareTensorBufferFunctor functor_;
};

ComputationOpHandle *GetUniquePendingComputationOpHandle(
    ShareTensorBufferOpHandle *share_tensor_op);

}  // namespace details
}  // namespace framework
}  // namespace paddle
