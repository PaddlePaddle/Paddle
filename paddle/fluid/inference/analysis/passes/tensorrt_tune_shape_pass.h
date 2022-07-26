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
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/inference/analysis/analysis_pass.h"
#include "paddle/phi/backends/dynload/port.h"

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
* 2. Make reuse plan: the vars can be reused if there is no overlap(on lifetime)
* between
* them.
* The final plan is a mapping table in which the key represents the original
* name of var and the value in the table represents the current name of var.
* 3. Perform reuse plan: Replace all var's name in the model according to the
* mapping table.
*/
class TensorrtTuneShapePass : public AnalysisPass {
 public:
  virtual ~TensorrtTuneShapePass() = default;

 protected:
  void RunImpl(Argument *argument) override;
  void StatisticShapeRangeInfo(std::map<std::string, std::vector<std::vector<int32_t>>> shape_info);
  void PrepareScope(Argument *argument, framework::Scope *scope);
  std::unique_ptr<framework::ProgramDesc> LoadModel(const std::string &path, framework::Scope *scope, const platform::Place &place);
  std::unique_ptr<framework::ProgramDesc> LoadModel(const std::string &program_path, const std::string &params_path, framework::Scope *scope, const platform::Place &place, bool model_from_memory);

 public:
  std::string repr() const override;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
