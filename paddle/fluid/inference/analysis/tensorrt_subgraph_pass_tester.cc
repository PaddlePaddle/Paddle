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

DEFINE_string(model_dir, "", "inference test model dir");

TEST(TensorRTSubGraph, single_pass) {
  auto desc = LoadProgramDesc();
  auto dfg = ProgramDescToDFG(desc);

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

  DFG_GraphvizDrawPass::Config config{"./", "test"};
  DFG_GraphvizDrawPass dfg_pass(config);
  dfg_pass.Initialize();

  DFG_GraphvizDrawPass dfg_pass1(config);
  dfg_pass1.Initialize();

  dfg_pass.Run(&dfg);

  TensorRTSubGraphPass trt_pass(std::move(teller));
  trt_pass.Initialize();

  trt_pass.Run(&dfg);

  dfg_pass1.Run(&dfg);

  // Check the TRT op's block desc
  for (auto node : dfg.nodes.nodes()) {
    if (node->IsFunctionBlock()) {
    }
  }
}

TEST(TensorRTSubGraph, pass_manager) {}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
