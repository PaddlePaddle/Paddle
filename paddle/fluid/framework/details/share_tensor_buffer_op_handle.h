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
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/op_handle_base.h"

namespace paddle {
namespace framework {

class Variable;
class Scope;
class Tensor;

namespace ir {
class MemOptVarInfo;
}  // namespace ir

namespace details {

class ShareTensorBufferOpHandle : public OpHandleBase {
 public:
  ShareTensorBufferOpHandle(ir::Node *node, const Scope *scope,
                            size_t scope_idx,
                            const std::vector<ir::MemOptVarInfo *> &in_vars,
                            const std::vector<std::string> &out_vars);

  std::unordered_set<std::string> ReusedVarSet() const;

  size_t GetScopeIdx() const { return scope_idx_; }

 protected:
  void RunImpl() final;

  virtual std::string MemoryReuseDebugString(size_t i) const = 0;

  Tensor *GetTensor(Scope **exec_scope, const std::string &name);

  const Scope *scope_;
  size_t scope_idx_;
  std::vector<ir::MemOptVarInfo *> in_vars_;
  std::vector<std::string> out_vars_;
  std::vector<bool> is_shared_;
};

class InplaceShareTensorBufferOpHandle : public ShareTensorBufferOpHandle {
 public:
  InplaceShareTensorBufferOpHandle(
      ir::Node *node, const Scope *scope, size_t scope_idx,
      const std::string &op_type,
      const std::vector<std::pair<std::string, std::string>> &in_out_params,
      const std::vector<ir::MemOptVarInfo *> &in_vars,
      const std::vector<std::string> &out_vars);

 protected:
  std::string Name() const override { return "inplace"; }

  std::string MemoryReuseDebugString(size_t i) const override;

 private:
  std::string op_type_;
  std::vector<std::pair<std::string, std::string>> in_out_params_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
