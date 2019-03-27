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

#include <algorithm>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

class MemoryOptimizePass : public ir::Pass {
 protected:
  ir::Graph* ApplyImpl(ir::Graph* graph) const override;
  // fill the variable map(var_nodes) by version.
  void InitSSAGraphNodes() const;

 private:
  // update program descs
  void RenameVarInGraphDesc(const std::string& var,
                            const std::string& cache_var, size_t idx) const;
  // update ir nodes
  void RenameVarInGraphNode(const std::string& var,
                            const std::string& cache_var, size_t idx,
                            ir::Graph* graph) const;

  void SubGraphOptimize(OpDesc* op_desc) const;
  // 1. scan op with subblock and collect the output/input vars.
  // while, while_grad, conditional_block
  // 2. scan distributed ops and collect the output/input vars
  void CollectSkipVarsSet(const std::unordered_set<ir::Node*>&) const;

 private:
  // Reuse Node Pool, Owned.
  mutable OrderedSet pool_;
  // controlflow Graph
  mutable std::unique_ptr<ControlFlowGraph> cfg_;
  // skip set
  mutable std::unordered_set<std::string> skip_set_;
  // var nodes
  mutable std::map<std::string, std::vector<ir::Node*>> var_nodes_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
