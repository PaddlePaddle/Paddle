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
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

SubGraphSplitter::NodeInsideSubgraphTeller teller = [](const Node* node) {
  if (node->type() != Node::Type::kFunction) return false;
  const auto* func = static_cast<const Function*>(node);
  if (func->func_type() == "elementwise_add" || func->func_type() == "relu" ||
      func->func_type() == "conv2d" || func->func_type() == "mul" ||
      func->func_type() == "sigmoid" || func->func_type() == "softmax") {
    LOG(INFO) << "sub-graph marked " << node->repr();
    return true;
  }
  return false;
};

TEST_F(DFG_Tester, Split) {
  auto desc = LoadProgramDesc();
  auto dfg = ProgramDescToDFG(desc);
  LOG(INFO) << "spliter\n" << dfg.DotString();

  ASSERT_GT(dfg.nodes.size(), 5UL);

  auto subgraphs = SubGraphSplitter(&dfg, teller)();

  // Check the number of the marked nodes.
  int marked_nodes = 0;
  for (auto& node : dfg.nodes.nodes()) {
    if (node->IsFunction() &&
        node->attr(SubGraphSplitter::kMarkerAttrName).Bool()) {
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

TEST_F(DFG_Tester, Fuse) {
  auto desc = LoadProgramDesc();
  auto dfg = ProgramDescToDFG(desc);

  size_t count0 = dfg.nodes.size();

  SubGraphFuse fuse(&dfg, teller);
  fuse();

  int count1 = 0;
  for (auto& node : dfg.nodes.nodes()) {
    if (node->deleted()) {
      LOG(INFO) << "deleted " << node->repr();
    }
    count1 += node->deleted();
  }

  // At least one nodes should be deleted.
  ASSERT_EQ(dfg.nodes.size(), count0 + 1);  // added a new FunctionBlock
  ASSERT_EQ(6UL, count1);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
