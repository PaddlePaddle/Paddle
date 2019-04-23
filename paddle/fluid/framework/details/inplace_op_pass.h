// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may abtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace details {

class GraphView {
 public:
  GraphView() = default;

  void Build(ir::Graph* g);

  const std::vector<ir::Node*>& AllOps();

  ir::Node* GetNodeByName(const std::string& name,
                          const std::vector<ir::Node*>& nodes) const;

  std::vector<ir::Node*> PendingOpsOnVar(ir::Node* var);

  // Will Deperated in the future.
  // NOTE(dzhwinter) :
  // 1. Python memory optimize will reuse
  // memory based var name, so different op output may
  // have the same variable name. enable inplace on such node
  // will generate a circle in ssa graph.
  // 2. DistributeTranspiler will use unique name to
  // map the parameter and gradient, must be skipped.
  bool InSkipSet(const std::string& var) const;

  bool CheckDeps(ir::Node* var, ir::Node* current_op) const;
  bool CheckOpDeps(ir::Node* op1, ir::Node* op2) const;
  void TopoSort(ir::Graph* g);

 private:
  std::vector<ir::Node*> ops_;
  std::unordered_set<std::string> skip_set_;  // mem opt affect nodes
  std::map<ir::Node*, std::unordered_set<ir::Node*>> adj_list_;
  std::unordered_map<ir::Node*, uint32_t> op_level_;
};

// swap pairs in sequence
typedef std::vector<std::pair<ir::Node*, ir::Node*>> NodeSwapQueue;
class InplacePass : public ir::Pass {
 public:
  InplacePass();

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  void InitSSAGraphNodes() const;

 private:
  const NodeSwapQueue TryInplaceModifyVar(const std::string& var,
                                          const std::string& cache_var,
                                          const size_t& idx,
                                          ir::Graph* graph) const;

  void CommitModify(const NodeSwapQueue&, ir::Graph* graph) const;

  void WithdrawModify(const NodeSwapQueue& nodes, ir::Graph* graph) const;

  void InplaceModifyDesc(const std::string& in_var, const std::string& out_var,
                         const size_t& idx) const;

  void TryInplaceOpInputOutput(ir::Node* op, ir::Graph* graph) const;

  mutable std::map<std::string, std::vector<ir::Node*>> var_nodes_;

  mutable std::unordered_set<std::string> whitelist_;
  mutable GraphView view_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
