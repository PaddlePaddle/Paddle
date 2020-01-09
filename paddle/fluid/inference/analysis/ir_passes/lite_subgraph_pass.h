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
#include <paddle/fluid/framework/ir/fuse_pass_base.h>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"

namespace paddle {
namespace inference {
namespace analysis {

class LiteSubgraphPass : public framework::ir::FusePassBase {
 public:
  void ApplyImpl(framework::ir::Graph* graph) const override;

 private:
  void BuildOperator(framework::ir::Node* merged_node,
                     framework::ProgramDesc* global_program,
                     std::vector<std::string>* repetitive_params) const;

  void SetUpEngine(framework::ProgramDesc* program,
                   const std::vector<std::string>& repetitive_params,
                   const std::string& unique_key,
                   bool dump_model = false) const;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
