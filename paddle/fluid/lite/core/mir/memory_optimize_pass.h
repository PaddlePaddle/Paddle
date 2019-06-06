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
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/lite/core/mir/pass.h"
#include "paddle/fluid/lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {

using lifecycle_t = std::pair<int, int>;

enum class MemoryOptimizeKind : int { kGreedy = 0, kAdapt, Num };

typedef struct {
  std::string name;
  size_t size;
  int cluster;
  lifecycle_t lifetime;
  std::unordered_set<std::string> adj;
} MemNode;

/*
 * Memory optimization pass for inference
 */
class MemoryOptimizePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  const MemoryOptimizeKind& memory_optimize_kind() {
    return memory_optimize_kind_;
  }
  MemoryOptimizeKind* mutable_memory_optimize_kind() {
    return &memory_optimize_kind_;
  }

 private:
  // Collect the lifecycles of the tensors.
  // Traverse the graph in topological order.
  // The traversal order also affect the lifecycles.
  void CollectLifeCycle(
      SSAGraph* graph,
      std::unordered_map<std::string, lifecycle_t>* lifecycles) const;

  // Collect the memory size of the tensors.
  void CollectVarMemorySize(
      SSAGraph* graph,
      std::unordered_map<std::string, size_t>* space_table) const;

  void CollectOverlapInfo(
      const std::unordered_map<std::string, std::pair<int, int>>& lifecycles,
      const std::unordered_map<std::string, size_t>& space_table,
      std::vector<MemNode>* mem_nodes);

  void MakeReusePlan(SSAGraph* graph,
                     std::unordered_map<std::string, std::string>* node2cluster,
                     std::vector<MemNode>* mem_nodes);

  // Greedy var reuse strategy
  void MemoryOptimizeGreedy(
      SSAGraph* graph,
      std::unordered_map<std::string, std::string>* node2cluster,
      std::vector<MemNode>* mem_nodes);

  // Update variables within scope for holder shared
  void UpdateScopeVarsByReuseTable(
      SSAGraph* graph,
      const std::unordered_map<std::string, std::string>& reuse_table) const;

  // Update SSAGraph info
  void UpdateSSAGraphByReuseTable(
      SSAGraph* graph,
      const std::unordered_map<std::string, std::string>& reuse_table) const;

  // Update var nodes
  void UpdateVarNodesByReuseTable(
      SSAGraph* graph,
      const std::unordered_map<std::string, std::string>& reuse_table) const;

  // Update OpDesc Input/Output information
  void UpdateOpNodesByReuseTable(
      SSAGraph* graph,
      const std::unordered_map<std::string, std::string>& reuse_table) const;

  bool IsVarCanBeReused(SSAGraph* graph, const std::string& name) const;

  void CollectLifeCycleHelper(
      SSAGraph* graph, int max_lifecycle,
      std::unordered_map<std::string, lifecycle_t>* lifecycles,
      const std::vector<std::string>& var_names) const;

 private:
  // The default var select strategy, this will be used in variable replacement
  MemoryOptimizeKind memory_optimize_kind_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
