/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/analysis/subgraph_splitter.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

SubgraphDetector::NodeInsideSubgraphTeller teller = [](const Node* node) {
  if (!node->IsOp()) return false;
  if (node->Op()->Type() == "elementwise_add" || node->Op()->Type() == "relu" ||
      node->Op()->Type() == "conv2d" || node->Op()->Type() == "mul" ||
      node->Op()->Type() == "sigmoid" || node->Op()->Type() == "softmax") {
    LOG(INFO) << "sub-graph marked " << node->Op()->Type();
    return true;
  }
  return false;
};

TEST(SubGraphSplitter, Split) {
  auto desc = LoadProgramDesc(FLAGS_inference_model_dir + "/__model__");
  auto dfg = ProgramDescToDFG(desc);
  LOG(INFO) << "spliter\n" << dfg.DotString();

  ASSERT_GT(dfg.nodes.size(), 5UL);

  auto subgraphs = SubgraphDetector(&dfg, teller)();

  // Check the number of the marked nodes.
  int marked_nodes = 0;
  for (auto& node : dfg.nodes.nodes()) {
    if (node->IsFunction() &&
        node->Get<bool>(SubgraphDetector::kMarkerAttrName).Bool()) {
      ++marked_nodes;
    }
  }
  EXPECT_EQ(marked_nodes, 6);

  // For human debug.
  for (auto& subgraph : subgraphs) {
    LOG(INFO) << "subgraph size " << subgraph.size();
    for (auto* node : subgraph) {
      LOG(INFO) << "node " << node->repr();
    }
  }

  ASSERT_EQ(subgraphs.size(), 1UL);
  // The last sub-graph has 5 Functions.
  ASSERT_EQ(subgraphs.back().size(), 6UL);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
