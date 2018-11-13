// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

// "Memory pool contains a lot unlived variables."
// "If these variable is reused in the future, it will be added to garbage"
// "collector (gc). Which will be cleared early than the scope destruction."
// "Enable it will tigger gc to the pool. default disabled."

class MemoryReusePass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  // void UpdateGraphAndScope(std::unordered_set<ir::Node*> *ir_nodes,
  //                          GraphOps* graph_ops,
  //                          GraphVars* graph_vars;
  //                              ReusedNodePairMap* reused_node_map) const;
  // replace reused pair in program and IR.
  void UpdateGraphAndDesc(ir::Node* op, const std::unordered_set<ir::Node*> *ir_nodes, ReusedNodePairMap* reused_node_map) const;
  // replace reused var_handle and op_handle
  void UpdateOpHandleAndScope(details::OpHandleBase* op, GraphOps* ops, GraphVars* vars, ReusedNodePairMap* reused_node_map) const;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
