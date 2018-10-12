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

#include "paddle/fluid/framework/ir/conv_bias_mkldnn_fuse_pass.h"

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
  } else if (type == "elementwise_add") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
  }
  op->SetOutput("Out", outputs);
}

// a->OP0->b
// b->OP1->c
// (c, weights)->conv->f
// (f, bias)->elementwise_add->g
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
  SetOp(&prog, "conv2d", std::vector<std::string>({"c", "weights"}),
        std::vector<std::string>({"f"}));
  SetOp(&prog, "elementwise_add", std::vector<std::string>({"f", "bias"}),
        std::vector<std::string>({"g"}));

  return prog;
}

TEST(ConvBiasFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("conv_bias_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: conv, elementwise_add, conv_out
  // Add 1 Node: ConvBias
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_bias op in newly generated graph
  int conv_bias_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      if (node->Op()->HasAttr("use_mkldnn")) {
        bool use_mkldnn = boost::get<bool>(node->Op()->GetAttr("use_mkldnn"));
        if (use_mkldnn) {
          auto names = node->Op()->InputNames();
          if (std::find(names.begin(), names.end(), "Bias") != names.end()) {
            conv_bias_count++;
          }
        }
      }
    }
  }
  EXPECT_EQ(conv_bias_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_bias_mkldnn_fuse_pass);
