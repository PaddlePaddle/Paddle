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
#include <unordered_map>
#include <utility>

#include "paddle/common/enforce.h"
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace analysis {

/* Memory optimization.
 * We will perform the following operation:
 * 1. Collect all var's lifetime.
 * 2. Make reuse plan: the vars can be reused if there is no overlap(on
 * lifetime) between them. The final plan is a mapping table in which the key
 * represents the original name of var and the value in the table represents the
 * current name of var.
 * 3. Perform reuse plan: Replace all var's name in the model according to the
 * mapping table.
 */
class MemoryOptimizePass : public AnalysisPass {
 public:
  using space_table_t = std::unordered_map<std::string, size_t>;
  using lifecycle_t = std::pair<int, int>;

  virtual ~MemoryOptimizePass() = default;

 protected:
  void RunImpl(Argument *argument) override;

 private:
  void CollectLifeCycle(
      framework::ir::Graph *graph,
      std::unordered_map<std::string, lifecycle_t> *lifecycles,
      int sort_kind) const;

  void CollectVarMemorySize(framework::ir::Graph *graph,
                            space_table_t *space_table) const;

 public:
  std::string repr() const override;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
