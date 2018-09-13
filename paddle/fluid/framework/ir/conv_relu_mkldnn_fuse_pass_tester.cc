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

#include "paddle/fluid/framework/ir/conv_relu_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", true);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
  } else if (type == "relu") {
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
}

// a->OP0->b
// b->OP1->c
// (c, weights, bias)->conv->f
// (f)->relu->g
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "weights", "bias", "f", "g"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}));
  SetOp(&prog, "conv2d", std::vector<std::string>({"c", "weights", "bias"}),
        std::vector<std::string>({"f"}));
  SetOp(&prog, "relu", std::vector<std::string>({"f"}),
        std::vector<std::string>({"g"}));

  return prog;
}

TEST(ConvReLUFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("conv_relu_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: CONV, RELU, conv_out
  // Add 1 Node: ConvReLU
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_relu op in newly generated graph
  int conv_relu_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      if (node->Op()->HasAttr("use_mkldnn")) {
        bool use_mkldnn = boost::get<bool>(node->Op()->GetAttr("use_mkldnn"));
        if (use_mkldnn) {
          if (node->Op()->HasAttr("fuse_relu")) {
            bool fuse_relu = boost::get<bool>(node->Op()->GetAttr("fuse_relu"));
            if (fuse_relu) {
              ++conv_relu_count;
            }
          }
        }
      }
    }
  }
  EXPECT_EQ(conv_relu_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_relu_mkldnn_fuse_pass);
