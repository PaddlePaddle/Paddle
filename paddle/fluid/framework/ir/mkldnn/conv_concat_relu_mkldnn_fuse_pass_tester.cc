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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/mkldnn/conv_activation_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           bool use_mkldnn = true) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetAttr("fuse_activation", std::string(""));
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2) {
      op->SetInput("Bias", {inputs[2]});
    }
    op->SetOutput("Output", outputs);
  } else if (type == "relu") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  } else if (type == "pool2d") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  } else if (type == "concat") {
    op->SetAttr("use_mkldnn", use_mkldnn);
    op->SetAttr("axis", 0);
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// (a1,w1)->conv1->c1
// (a2,w2,b2)->conv2->c2
// if put_only_convs_before_concat=true
//   (a3,w3)->conv3->c3
// else
//   a3->pool1->c3
//
// (c1,c2,c3)->concat1->d
// d->relu1->e
ProgramDesc BuildProgramDesc(bool put_only_convs_before_concat,
                             bool all_convs_use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : std::initializer_list<std::string>({"a1",
                                                     "w1",
                                                     "c1",
                                                     "a2",
                                                     "w2",
                                                     "b2",
                                                     "c2",
                                                     "a3",
                                                     "w3",
                                                     "c3",
                                                     "d",
                                                     "e"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v.find("w") == 0 || v.find("b") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "conv2d", {"a1", "w1", "b1"}, {"c1"}, all_convs_use_mkldnn);
  SetOp(&prog, "conv2d", {"a2", "w2", "b2"}, {"c2"});
  if (put_only_convs_before_concat) {
    SetOp(&prog, "conv2d", {"a3", "w3", "b3"}, {"c3"});
  } else {
    SetOp(&prog, "pool2d", {"a3"}, {"c3"});
  }
  SetOp(&prog, "concat", {"c1", "c2", "c3"}, {"d"});
  SetOp(&prog, "relu", {"d"}, {"e"});

  return prog;
}

void MainTest(const ProgramDesc& prog, bool fuse_relu) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  int original_nodes_num = graph->Nodes().size();

  auto pass = PassRegistry::Instance().Get("conv_activation_mkldnn_fuse_pass");
  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();

  if (fuse_relu) {
    // Remove 2 nodes: concat_out, relu
    EXPECT_EQ(original_nodes_num - 2, current_nodes_num);
  } else {
    EXPECT_EQ(original_nodes_num, current_nodes_num);
  }

  int relu_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "conv2d") {
        ASSERT_TRUE(op->HasAttr("fuse_activation"));
        bool fuse_relu_attr =
            (PADDLE_GET_CONST(std::string, op->GetAttr("fuse_activation")) ==
             "relu");
        EXPECT_EQ(fuse_relu, fuse_relu_attr);
      } else if (op->Type() == "relu") {
        relu_count++;
      }
    }
  }
  EXPECT_EQ(relu_count, fuse_relu ? 0 : 1);
}

TEST(ConvConcatReLUFusePass, only_convs_before_concat) {
  bool all_convs_use_mkldnn = true;
  bool put_only_convs_before_concat = true;
  auto prog =
      BuildProgramDesc(put_only_convs_before_concat, all_convs_use_mkldnn);

  bool expect_relu_fuse = true;
  MainTest(prog, expect_relu_fuse);
}

TEST(ConvConcatReLUFusePass, only_convs_before_concat_but_one_non_mkldnn) {
  bool all_convs_use_mkldnn = false;
  bool put_only_convs_before_concat = true;
  auto prog =
      BuildProgramDesc(put_only_convs_before_concat, all_convs_use_mkldnn);

  bool expect_relu_fuse = false;
  MainTest(prog, expect_relu_fuse);
}

TEST(ConvConcatReLUFusePass, convs_and_pool_before_concat) {
  bool all_convs_use_mkldnn = true;
  bool put_only_convs_before_concat = false;
  auto prog =
      BuildProgramDesc(put_only_convs_before_concat, all_convs_use_mkldnn);

  bool expect_relu_fuse = false;
  MainTest(prog, expect_relu_fuse);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_activation_mkldnn_fuse_pass);
