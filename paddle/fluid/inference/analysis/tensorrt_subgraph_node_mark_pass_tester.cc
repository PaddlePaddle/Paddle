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

#include "paddle/fluid/inference/analysis/tensorrt_subgraph_node_mark_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/node_attr_flags.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

TEST_F(DFG_Tester, tensorrt_subgraph_node_mark_pass) {
  // init
  FluidToDataFlowGraphPass pass;
  ASSERT_TRUE(pass.Initialize(&argument));
  argument.main_dfg.reset(new DataFlowGraph);
  pass.Run(argument.main_dfg.get());

  TensorRTSubgraphNodeMarkPass::teller_t teller = [](const Node* node) {
    return node->IsFunction() &&
           static_cast<const Function*>(node)->func_type() == "mul";
  };
  TensorRTSubgraphNodeMarkPass pass1(teller);
  ASSERT_TRUE(pass1.Initialize(&argument));
  pass1.Run(argument.main_dfg.get());

  int counter{0};
  for (auto& node : argument.main_dfg->nodes.nodes()) {
    counter += node->attr(ATTR_supported_by_tensorrt).Bool();
  }

  LOG(INFO) << counter << " nodes marked";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
