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

#include "paddle/fluid/framework/ir/mkldnn/conv_activation_mkldnn_fuse_pass.h"

#include <gtest/gtest.h>
#include <vector>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool is_activation = false,
           bool use_mkldnn = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("name", name);
  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetAttr("groups", 1);
    op->SetAttr("padding_algorithm", std::string("EXPLICIT"));
    op->SetAttr("data_format", std::string("NCHW"));
    op->SetAttr("strides", std::vector<int>({1, 1}));
    op->SetAttr("dilations", std::vector<int>({1, 1}));
    op->SetAttr("paddings", std::vector<int>({0, 0}));
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
    op->SetOutput("Output", outputs);
  } else if (is_activation) {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", inputs);
    if (type == "leaky_relu") {
      op->SetAttr("alpha", 0.02f);
    } else if (type == "relu6") {
      op->SetAttr("threshold", 6.0f);
    } else if (type == "swish") {
      op->SetAttr("beta", 1.0f);
    }
    op->SetOutput("Out", outputs);
  }

  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// a->OP0->b
// b->OP1->c
// (c, weights, bias)->conv->f
// (f)->activation->g
ProgramDesc BuildProgramDesc(std::string activation) {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "weights", "bias", "f", "g",
                                 "h", "weights2", "bias2", "k", "l", "m"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias" || v == "weights2" || v == "bias2") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", "op0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", "op1", std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}));
  // conv+activation, both with MKL-DNN
  SetOp(&prog, "conv2d", "conv1",
        std::vector<std::string>({"c", "weights", "bias"}),
        std::vector<std::string>({"f"}), false, true);
  SetOp(&prog, activation, "activation1", std::vector<std::string>({"f"}),
        std::vector<std::string>({"g"}), true, true);
  SetOp(&prog, "OP3", "op3", std::vector<std::string>({"g"}),
        std::vector<std::string>({"h"}));
  // conv+activation, only one with MKL-DNN
  SetOp(&prog, "conv2d", "conv2",
        std::vector<std::string>({"h", "weights2", "bias2"}),
        std::vector<std::string>({"k"}), false, true);
  SetOp(&prog, "activation", "activation2", std::vector<std::string>({"k"}),
        std::vector<std::string>({"l"}), true, false);
  SetOp(&prog, "OP4", "op4", std::vector<std::string>({"l"}),
        std::vector<std::string>({"m"}));

  return prog;
}

void MainTest(std::string activation) {
  auto prog = BuildProgramDesc(activation);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass =
      PassRegistry::Instance().Get("conv_" + activation + "_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: CONV, activation, conv_out
  // Add 1 Node: ConvActivation
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_activation op in newly generated graph
  int conv_activation_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
      if (op->GetAttrIfExists<std::string>("fuse_activation") == activation) {
        ++conv_activation_count;
      }
      // check if only "conv1" convolution is fused
      if (op_name == "conv1") {
        ASSERT_TRUE(op->HasAttr("fuse_activation"));
      } else if (op_name == "conv2") {
        ASSERT_FALSE(op->HasAttr("fuse_activation"));
      }
    }
  }
  EXPECT_EQ(conv_activation_count, 1);
}

TEST(ConvActivationFusePass, conv_relu_fuse_pass) { MainTest("relu"); }
TEST(ConvActivationFusePass, conv_leaky_relu_fuse_pass) {
  MainTest("leaky_relu");
}
TEST(ConvActivationFusePass, conv_relu6_fuse_pass) { MainTest("relu6"); }
TEST(ConvActivationFusePass, conv_swish_fuse_pass) { MainTest("swish"); }
TEST(ConvActivationFusePass, conv_hard_swish_fuse_pass) {
  MainTest("hard_swish");
}
TEST(ConvActivationFusePass, conv_mish_fuse_pass) { MainTest("mish"); }
TEST(ConvActivationFusePass, conv_hard_sigmoid_fuse_pass) {
  MainTest("hard_sigmoid");
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_activation_mkldnn_fuse_pass);
