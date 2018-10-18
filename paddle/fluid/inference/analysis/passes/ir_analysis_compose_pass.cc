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

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/inference/analysis/ir_pass_manager.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/inference/analysis/passes/ir_analysis_compose_pass.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

void IrAnalysisComposePass::RunImpl(Argument *argument) {
  ARGUMENT_CHECK_FIELD(argument, ir_analysis_passes);
  CreateIrPasses(argument, *argument->ir_analysis_passes());
  InitTensorRTAttrs(argument);
  ApplyIrPasses(argument);
}

std::string IrAnalysisComposePass::repr() const {
  return "ir-analysis-compose-pass";
}

void IrAnalysisComposePass::CreateIrPasses(
    Argument *argument, const std::vector<std::string> &passes) {
  std::string pre_pass;
  int pass_num = 0;
  for (const std::string &pass_name : passes) {
    string::PrettyLogEndl(string::Style::H2(), "--- Running IR pass [%s]",
                          pass_name);
    auto pass = framework::ir::PassRegistry::Instance().Get(pass_name);

    // Set some pass attributes.
    if (pass_name == "ir_analysis_pass") {
      pass->Set("tensorrt_node_teller",
                new SubgraphDetector::NodeInsideSubgraphTeller(
                    *argument->tensorrt_node_teller()));
    }

    if (pass_name == "graph_viz_pass") {
      std::string dot_file_path = std::to_string(pass_num) + "_ir_" +
                                  (pre_pass.empty() ? "origin" : pre_pass) +
                                  ".dot";
      pass->Set("graph_viz_path", new std::string(std::move(dot_file_path)));
      pass_num++;
    }

    // graph_ = pass->Apply(std::move(graph_));
    pre_pass = pass_name;
  }
}

void IrAnalysisComposePass::InitTensorRTAttrs(Argument *argument) {
  if (argument->use_tensorrt() && *argument->use_tensorrt()) {
    argument->SetTensorRtNodeTeller([](const framework::ir::Node *node) {
      std::unordered_set<std::string> teller_set(
          {"mul", "conv2d", "pool2d", "relu", "softmax", "sigmoid",
           "depthwise_conv2d", "batch_norm", "concat", "tanh", "pad",
           "elementwise_add", "dropout"});
      if (!node->IsOp()) return false;

      if (teller_set.count(node->Op()->Type())) {
        return true;
      } else {
        return false;
      }
    });
  }
}

void IrAnalysisComposePass::ApplyIrPasses(Argument *argument) {
  // Create passes.
  IRPassManager manager(argument);
  manager.Apply();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
