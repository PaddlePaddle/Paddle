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

#include "paddle/fluid/inference/analysis/tensorrt_subgraph_pass.h"

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/ut_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

DEFINE_string(dot_dir, "./", "");

TEST_F(DFG_Tester, tensorrt_single_pass) {
  std::unordered_set<std::string> teller_set(
      {"elementwise_add", "mul", "sigmoid"});
  SubGraphSplitter::NodeInsideSubgraphTeller teller = [&](const Node* node) {
    if (node->type() != Node::Type::kFunction) return false;
    const auto* func = static_cast<const Function*>(node);
    if (teller_set.count(func->func_type())) return true;
    return false;
  };

  LOG(INFO) << "init";
  DFG_GraphvizDrawPass::Config config{FLAGS_dot_dir, "origin"};
  DFG_GraphvizDrawPass::Config config1{FLAGS_dot_dir, "fusion"};

  DFG_GraphvizDrawPass dfg_pass(config);
  DFG_GraphvizDrawPass dfg_pass1(config1);
  FluidToDataFlowGraphPass pass0;
  TensorRTSubGraphPass trt_pass(std::move(teller));

  LOG(INFO) << "Initialize";
  dfg_pass.Initialize(&argument);
  dfg_pass1.Initialize(&argument);
  pass0.Initialize(&argument);
  trt_pass.Initialize(&argument);

  LOG(INFO) << "Run";
  argument.main_dfg.reset(new DataFlowGraph);
  pass0.Run(argument.main_dfg.get());
  dfg_pass.Run(argument.main_dfg.get());
  trt_pass.Run(argument.main_dfg.get());
  dfg_pass1.Run(argument.main_dfg.get());

  // Check the TRT op's block desc
  for (auto& node : argument.main_dfg->nodes.nodes()) {
    if (node->IsFunctionBlock()) {
      LOG(INFO) << "get function block";
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
