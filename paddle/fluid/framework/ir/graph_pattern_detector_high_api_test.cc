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

#include "paddle/fluid/framework/ir/graph_pattern_detector_high_api.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

// An demo.
class FcFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // create nodes.
    auto& x = VarNode("x");
    auto& W = VarNode("W");
    auto& b = VarNode("b");
    auto& mul = OpNode("mul", "mul");
    auto& mul_out = VarNode("mul_out");
    auto& add = OpNode("add", "elementwise_add");
    auto& Out = VarNode("Out");

    // create topology.
    std::vector<PDNode2>({W, x}) >> mul >> mul_out;
    std::vector<PDNode2>({mul_out, b}) >> add >> Out;

    // Some op specialities.
    mul_out.pd_node().AsIntermediate();
    mul.pd_node().AsIntermediate();
    add.pd_node().AsIntermediate();
  }

  void InsertNewNode(ir::Graph* graph, const key2nodes_t& matched) override {
    auto op_desc = GenOpDesc(matched);
    auto* new_op_node = graph->CreateOpNode(&op_desc);

    IR_NODE_LINK_TO(matched.at("W"), new_op_node);
    IR_NODE_LINK_TO(matched.at("x"), new_op_node);
    IR_NODE_LINK_TO(matched.at("b"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
  }

 private:
  OpDesc GenOpDesc(const key2nodes_t& matched) override {
    framework::OpDesc op_desc;
    op_desc.SetType("fc");
    op_desc.SetInput("Input", {matched.at("x")->Name()});
    op_desc.SetInput("W", {matched.at("W")->Name()});
    op_desc.SetInput("Bias", {matched.at("b")->Name()});
    op_desc.SetOutput("Out", {matched.at("Out")->Name()});
    return op_desc;
  }
};

std::unique_ptr<Graph> BuildGraph() {
  ProgramDesc program_desc;
  auto* main_block = program_desc.MutableBlock(0);

  auto* mul = main_block->AppendOp();
  auto* add = main_block->AppendOp();
  auto* scale = main_block->AppendOp();
  main_block->Var("x");
  main_block->Var("b");
  main_block->Var("mul_out");
  main_block->Var("w");
  main_block->Var("out");
  main_block->Var("out1");

  mul->SetInput("X", {"x"});
  mul->SetInput("Y", {"w"});
  mul->SetOutput("Out", {"mul_out"});
  mul->SetType("mul");

  add->SetInput("X", {"mul_out"});
  add->SetInput("Y", {"b"});
  add->SetOutput("Out", {"out"});
  add->SetType("elementwise_add");

  scale->SetInput("X", {"out"});
  scale->SetOutput("Out", {"out1"});

  program_desc.Flush();

  return std::unique_ptr<Graph>(new Graph(program_desc));
}

TEST(graph_pattern_detector2, graph_test) {
  auto graph = BuildGraph();
  ASSERT_EQ(graph->Nodes().size(), 9UL);
  auto pass = PassRegistry::Instance().Get("graph_viz_pass");
  pass->Set("graph_viz_path", new std::string("./1.dot"));
  pass->Apply(graph.get());
}

TEST(graph_pattern_detector2, test) {
  auto graph = BuildGraph();
  FcFuser fuser;
  fuser(graph.get());
  ASSERT_EQ(graph->Nodes().size(), 9 - 3 + 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(graph_viz_pass);
