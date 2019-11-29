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
#include "paddle/fluid/inference/anakin/engine.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_util.h"

using anakin::Precision;
using anakin::saber::NV;
namespace paddle {
namespace inference {
namespace analysis {

class AnakinSubgraphPass : public framework::ir::FusePassBase {
 public:
  void ApplyImpl(framework::ir::Graph *graph) const override;

 private:
  void CreateAnakinOp(framework::ir::Node *x, framework::ir::Graph *graph,
                      const std::vector<std::string> &graph_params,
                      std::vector<std::string> *repetitive_params) const;
  void CleanIntermediateOutputs(framework::ir::Node *node);
  template <::anakin::Precision PrecisionT>
  void CreateAnakinEngine(framework::BlockDesc *block_desc,
                          const std::vector<std::string> &params,
                          const std::set<std::string> &input_names,
                          const std::vector<std::string> &output_mapping,
                          const std::vector<std::string> &program_inputs,
                          const std::string &engine_key) const;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
