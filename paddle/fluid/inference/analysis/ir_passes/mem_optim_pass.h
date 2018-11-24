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
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace inference {
namespace analysis {

class MemOptimPass : public framework::ir::Pass {
 public:
  virtual ~MemOptimPass() = default;

 protected:
  std::unique_ptr<framework::ir::Graph> ApplyImpl(
      std::unique_ptr<framework::ir::Graph> graph) const;

 private:
  using lifecycle_t = std::pair<int, int>;
  void CollectLifeCycle(
      std::unordered_map<std::string, lifecycle_t>* lifecycles) const;

  void CollectShapes(
      std::unordered_map<std::string, framework::ir::Node*>* tensor_nodes,
      std::unordered_map<std::string, int>* space_table) const;

  void MakeReusePlan(const std::vector<std::unordered_set<std::string>> &var_clusters,
                       std::unordered_map<std::string, std::string> *reuse_table) const;

  void PerformReusePlan(
      const std::unordered_map<std::string, std::string>& reuse_table) const;

  mutable framework::ir::Graph* graph_{nullptr};
  mutable int max_lifecycle_{-1};
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
