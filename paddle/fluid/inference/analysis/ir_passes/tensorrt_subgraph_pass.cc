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

#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"
#include <vector>
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

std::unique_ptr<framework::ir::Graph> analysis::TensorRtSubgraphPass::ApplyImpl(
    std::unique_ptr<framework::ir::Graph> graph) const {
  SubgraphDetector::NodeInsideSubgraphTeller teller = [](const Node* x) {
    return true;
  };

  std::vector<std::vector<framework::ir::Node*>> subgraphs =
      SubgraphDetector(graph.get(), teller)();

  for (auto& subgraph : subgraphs) {
    LOG(INFO) << "detect " << subgraph.size() << " subgraphs";
  }

  return graph;
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("tensorrt_node_teller");
