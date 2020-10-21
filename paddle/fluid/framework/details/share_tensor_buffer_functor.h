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
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"

namespace paddle {
namespace framework {
class Scope;
namespace ir {
class MemOptVarInfo;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

// NOTE(paddle-dev): ShareTensorBufferFunctor is responsible for
// performing memory reuse in run-time. ShareTensorBufferOpHandle
// is only a wrapper of ShareTensorBufferFunctor.
// Once we find the run-time memory reuse strategy is time-consuming in
// scheduling, we should need a pass to move ShareTensorBufferFunctor into
// each ComputationOpHandle. ShareTensorBufferFunctor is preserved for
// this probable movement.
class ShareTensorBufferFunctor {
 public:
  ShareTensorBufferFunctor(
      Scope *scope, size_t scope_idx, const std::string &op_type,
      const std::vector<const ir::MemOptVarInfo *> &in_var_infos,
      const std::vector<std::string> &out_var_names, bool share_dims = false);

  void AddReuseVarPair(const ir::MemOptVarInfo *in_var_info,
                       const std::string &out_var_name);

  void SetShareDims(bool share_dims) { share_dims_ = share_dims; }

  void operator()(Scope *exec_scope);

  std::unordered_map<std::string, std::string> ReusedVars() const;

  size_t GetScopeIdx() const { return scope_idx_; }

  Scope *GetScope() { return scope_; }

 private:
  void CallOnce();

 private:
  Scope *scope_;
  Scope *exec_scope_{nullptr};

  size_t scope_idx_;
  std::string op_type_;
  std::vector<const ir::MemOptVarInfo *> in_var_infos_;
  std::vector<std::string> out_var_names_;

  std::vector<std::pair<const Variable *, Variable *>> in_out_vars_;

  // NOTE(zhiqiu): In the case of inplace addto, if the operator of
  // the in_out_vars is skipped during running, we should set the dims of output
  // as the same as input.
  bool share_dims_{false};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
