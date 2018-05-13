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

TEST_F(DFG_Tester, Split) {
  auto desc = LoadProgramDesc();
  auto dfg = ProgramDescToDFG(desc);
  LOG(INFO) << "spliter\n" << dfg.DotString();

  DataFlowGraph new_graph;
  SubGraphSplitter::NodeInsideSubgraphTeller teller = [](const Node* node) {
    LOG(INFO) << "inside teller";
    LOG(INFO) << "node.type " << static_cast<int>(node->type());
    if (node->type() != Node::Type::kFunction) return false;
    const auto* func = static_cast<const Function*>(node);
    LOG(INFO) << "func.type " << func->func_type();
    if (func->func_type() == "elementwise_sub") {
      return true;
    }
    return false;
  };
  ASSERT_GT(dfg.nodes.size(), 5);
  auto subgraphs = SubGraphSplitter(&dfg, teller)();
  LOG(INFO) << "subgraph size: " << subgraphs.size();
  ASSERT_EQ(subgraphs.size(), 1);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
