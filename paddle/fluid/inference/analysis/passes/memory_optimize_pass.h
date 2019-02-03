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
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Memory optimization pass for inference with pre-analysis of memory usage
 * without GC.
 * Different from training, the inference memory reuse strategies doesn't
 * include GC for that overhead is too much when batch size equals one.
 *
 * The inference memory reuse tries to pre-determine the tensor reusing strategy
 * without runtime overhead.
 *
 * To improve the strategy's performance, a warm-up running is introduced:
 *   - Before officially deploy the inference program, one should warm it up and
 *     generate some runtime cache,
 *   - Run the inference program with several batches of data, it will persist
 *     some runtime information about memory of tensors to disk, we call the
 *     information the memory reusing cache,
 *   - With the memory reusing cache, user can deploy the inference to a
 *     service, before running the model, the inference program will load the
 *     memory cache, analysis it and generate the best memory reusing strategy,
 *     and adjust the execution of the network.
 *
 * With the warm-up and memory reusing cache design, the memory reusing
 * algorithm can analysis the real memory consume of the tensors, even with the
 * flexible LoDTensor and special shape changing operators such as
 * sequence-pooling.
 */
class MemoryOptimizePass : public AnalysisPass {
 public:
  using space_table_t = std::unordered_map<std::string, size_t>;
  using lifecycle_t = std::pair<int, int>;

  struct MemoryAllocation {
    size_t allocated;  // allocated memory in byte.
    size_t saved;      // saved memory in byte.
    int sort_kind;     // the kind of the corresponding sorting algorithm.

    // Get the memory saving ratio of temporary variables.
    float GetSavingRatio() const;
  };

  virtual ~MemoryOptimizePass() = default;

 protected:
  void RunImpl(Argument *argument) override;

 private:
  void CollectLifeCycle(
      std::unordered_map<std::string, lifecycle_t> *lifecycles,
      int sort_kind) const;

  void CollectVarMemorySize(
      const std::unordered_map<std::string, size_t> &batch_var_ave_dim,
      std::unordered_map<std::string, framework::ir::Node *> *tensor_nodes,
      space_table_t *space_table) const;

  // Returns percentage of saved memory.
  void MakeReusePlan(
      const std::vector<std::unordered_set<std::string>> &var_clusters,
      const std::unordered_map<std::string, size_t> &var_batch_ave_size,
      const space_table_t &space_table,
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
