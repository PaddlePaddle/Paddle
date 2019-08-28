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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/inference/lite/op_teller.h"

#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/string/pretty_log.h"


namespace paddle {
namespace inference {
namespace analysis {

void analysis::LiteSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("lite_subgraph_pass", graph);

  // auto &lite_ops_filter = Get<std::vector<std::string>>("lite_ops_filter");
  std::vector<std::string> lite_ops_filter = {};

  auto teller = [&lite_ops_filter](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op())
      return false;
    else if (std::find(lite_ops_filter.begin(), lite_ops_filter.end(),
                       node->Op()->Type()) != lite_ops_filter.end())
      return false;
    return lite::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph, teller, 0 /* min_subgraph_size */);
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());

  // those parameter already exist in anakin, and should not have another copy
  // in fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    LOG(INFO) << "[lite_subgraph_pass] node name = " << node->Name();
    /*
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateAnakinOp(node, graph, graph_param_names, &repetitive_params);
      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
    */
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}



}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(lite_subgraph_pass,
              paddle::inference::analysis::LiteSubgraphPass);
