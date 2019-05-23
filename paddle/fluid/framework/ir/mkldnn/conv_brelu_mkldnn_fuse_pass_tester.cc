// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_brelu_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
  } else if (type == "relu6") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    if (use_mkldnn) {
      op->SetAttr("threshold", 6.0f);
    }
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// a->OP0->b
// b->OP1->c
// (c, weights, bias)->conv->f
// (f)->brelu->g
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "weights", "bias", "f", "g",
                                 "h", "weights2", "bias2", "k", "l"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", "op0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", "op1", std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}));
  // conv+brelu, both with MKL-DNN
  SetOp(&prog, "conv2d", "conv1",
        std::vector<std::string>({"c", "weights", "bias"}),
        std::vector<std::string>({"f"}), true);
  SetOp(&prog, "relu6", "relu1", std::vector<std::string>({"f"}),
        std::vector<std::string>({"g"}), true);
  SetOp(&prog, "OP3", "op3", std::vector<std::string>({"g"}),
        std::vector<std::string>({"h"}));
  // conv+brelu, only one with MKL-DNN
  SetOp(&prog, "conv2d", "conv2",
        std::vector<std::string>({"h", "weights2", "bias2"}),
        std::vector<std::string>({"k"}), true);
  SetOp(&prog, "relu6", "relu2", std::vector<std::string>({"k"}),
        std::vector<std::string>({"l"}));

  return prog;
}

TEST(ConvBReLUFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("conv_brelu_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: CONV, BRELU, conv_out
  // Add 1 Node: ConvBReLU
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_brelu op in newly generated graph
  int conv_brelu_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(boost::get<bool>(op->GetAttr("use_mkldnn")));
      // check if only "conv1" convolution is fused
      auto op_name = boost::get<std::string>(op->GetAttr("name"));
      if (op_name == "conv1") {
        ASSERT_TRUE(op->HasAttr("fuse_brelu"));
        ASSERT_TRUE(op->HasAttr("fuse_brelu_threshold"));

        bool fuse_brelu = boost::get<bool>(op->GetAttr("fuse_brelu"));
        if (fuse_brelu) {
          ++conv_brelu_count;
          float fuse_brelu_threshold =
              boost::get<float>(op->GetAttr("fuse_brelu_threshold"));
          EXPECT_EQ(fuse_brelu_threshold, 6.0f);
        }
      } else if (op_name == "conv2") {
        ASSERT_FALSE(op->HasAttr("fuse_brelu"));
      }
    }
  }
  EXPECT_EQ(conv_brelu_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_brelu_mkldnn_fuse_pass);
