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
#include <vector>
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/details/share_tensor_buffer_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/reference_count_pass_helper.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class MemoryReusePass : public Pass {
 protected:
  void ApplyImpl(Graph *graph) const final;

  virtual void Run(Graph *graph) const = 0;

  virtual std::string ReuseType() const = 0;

  bool TryReuseVar(details::VarHandle *in_var,
                   details::VarHandle *out_var) const;

  std::unordered_set<ir::Node *> FindNodesByName(
      const std::string &name, const std::vector<ir::Node *> &nodes) const;

  size_t ScopeNum() const { return all_vars_->size(); }

 private:
  VarDesc *GetVarDesc(details::VarHandle *var) const;

  bool IsVarsReusable(details::VarHandle *in_var,
                      details::VarHandle *out_var) const;

  bool IsVarAlreadyReused(details::VarHandle *var) const;

  details::ShareTensorBufferOpHandle *InsertShareTensorBufferOpHandleToGraph(
      details::ComputationOpHandle *op) const;

  details::VarHandle *InsertNewVarHandleBefore(details::VarHandle *var) const;

  void CollectShareTensorBufferOpHandles() const;

  void CollectReusedVars() const;

  void AddReuseVar(details::ComputationOpHandle *op, details::VarHandle *in_var,
                   details::VarHandle *out_var) const;

  void UpdateLastLiveOpOfVar(details::ComputationOpHandle *op,
                             details::VarHandle *in_var,
                             details::VarHandle *out_var) const;

 private:
  mutable Graph *graph_;
  mutable details::GraphVars *all_vars_;
  mutable MemOptVarInfoMapList *var_infos_;
  mutable std::vector<LastLiveOpsOfVars> *last_live_ops_of_vars_;

  mutable std::unordered_map<details::ComputationOpHandle *,
                             details::ShareTensorBufferOpHandle *>
      ops_;

  mutable std::vector<std::unordered_set<std::string>> reused_var_names_;

  mutable std::vector<std::unordered_map<std::string, VarDesc *>> var_descs_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
