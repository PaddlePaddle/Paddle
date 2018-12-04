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
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/fluid/inference/analysis/passes/memory_optimize_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Memory optimization pass for inference with pre-analysis of memory usage
 * without GC.
 * It should work with both the LoD or Non-LoD models.
*/
class MemoryOptimizePass : public AnalysisPass {
 public:
  struct MemoryAllocation {
    long long allocated;
    long long saved;
    float saving_ratio;
    int sort_kind;
  };

  virtual ~MemoryOptimizePass() = default;

 protected:
  void RunImpl(Argument *argument) override;

 private:
  using lifecycle_t = std::pair<int, int>;
  void CollectLifeCycle(
      std::unordered_map<std::string, lifecycle_t> *lifecycles,
      int sort_kind) const;

  void CollectShapes(
      std::unordered_map<std::string, framework::ir::Node *> *tensor_nodes,
      std::unordered_map<std::string, int> *space_table) const;

  // Returns percentage of saved memory.
  void MakeReusePlan(
      const std::vector<std::unordered_set<std::string>> &var_clusters,
      const std::unordered_map<std::string, size_t> &var_batch_ave_size,
      std::unordered_map<std::string, std::string> *reuse_table, int sort_kind,
      MemoryAllocation *memory_allocation) const;

  void PerformReusePlan(
      const std::unordered_map<std::string, std::string> &reuse_table,
      int sort_kind, std::unordered_set<std::string> *vars2remove) const;

 public:
  std::string repr() const override;

 private:
  mutable framework::ir::Graph *graph_{nullptr};
  mutable int max_lifecycle_{-1};
};

static std::string GetMemoryCachePath(const std::string &model_path,
                                      const std::string &prog_path) {
  auto path = model_path.empty() ? prog_path : model_path;
  return path + ".memory_cache";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
