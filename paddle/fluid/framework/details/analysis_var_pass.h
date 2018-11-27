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

#include <memory>
#include <set>
#include <string>

#include "paddle/fluid/framework/details/cfg_graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

class AnalysisVarPass : public ir::Pass {
 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(
      std::unique_ptr<ir::Graph> graph) const override;

 private:
  // update program descs
  void RenameVarInGraphDesc(const std::string& var,
                            const std::string& cache_var, size_t idx) const;
  // update ir nodes
  void RenameVarInGraphNode(const std::string& var,
                            const std::string& cache_var, size_t idx) const;
  // valid a tensor can be reuse or not
  bool NodeCanReused(ir::Node* node) const;
  // scan subblock and collect the output variables.
  std::unordered_set<ir::Node*> GetSubBlockOutputVars(
      const std::unordered_set<ir::Node*>&) const;
  // scan subblock and collect the output/input variables.
  std::unordered_set<std::string> GetSubBlockVars(
      const std::unordered_set<ir::Node*>&) const;
  // check op has subblock or not
  bool OpHasSubBlock(OpDesc* desc) const;

  // Reuse Node Pool, Owned.
  mutable details::OrderedNodePairPool pool_;
  // controlflow Graph
  mutable details::ControlFlowGraph cfg_;
  // skip set
  mutable std::unordered_set<std::string> skip_set_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
