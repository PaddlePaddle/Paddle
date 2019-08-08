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

#include "paddle/fluid/lite/core/mir/pattern_matcher_high_api.h"
#include <gtest/gtest.h>
#include <memory>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/compatible_tensor.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/program.h"

namespace paddle {
namespace lite {
namespace mir {

// An demo.
class FcFuser : public FuseBase {
 public:
  void BuildPattern() override {
    // create nodes.
    auto* x = VarNode("x")->assert_is_op_input("mul", "X");
    auto* W = VarNode("W")->assert_is_op_input("mul", "Y");
    auto* b = VarNode("b");
    auto* mul = OpNode("mul", "mul");
    auto* mul_out = VarNode("mul_out");
    auto* add = OpNode("add", "elementwise_add");
    auto* Out = VarNode("Out");

    // create topology.
    std::vector<PMNode*> mul_inputs{W, x};
    std::vector<PMNode*> add_inputs{mul_out, b};
    mul_inputs >> *mul >> *mul_out;
    add_inputs >> *add >> *Out;

    // Some op specialities.
    mul_out->AsIntermediate();
    mul->AsIntermediate();
    add->AsIntermediate();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto op_desc = GenOpDesc(matched);
    auto fc_op = LiteOpRegistry::Global().Create("fc");
    auto mul = matched.at("mul")->stmt()->op();
    auto* scope = mul->scope();
    auto& valid_places = mul->valid_places();
    fc_op->Attach(op_desc, scope);

    auto* new_op_node = graph->GraphCreateInstructNode(fc_op, valid_places);

    IR_NODE_LINK_TO(matched.at("W"), new_op_node);
    IR_NODE_LINK_TO(matched.at("x"), new_op_node);
    IR_NODE_LINK_TO(matched.at("b"), new_op_node);
    IR_NODE_LINK_TO(new_op_node, matched.at("Out"));
  }

 private:
  cpp::OpDesc GenOpDesc(const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("fc");
    op_desc.SetInput("Input", {matched.at("x")->arg()->name});
    op_desc.SetInput("W", {matched.at("W")->arg()->name});
    op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("Out")->arg()->name});
    op_desc.SetAttr("in_num_col_dims", 1);
    return op_desc;
  }
};

std::unique_ptr<SSAGraph> BuildGraph(framework::ProgramDesc* program_desc,
                                     const std::shared_ptr<Scope>& scope,
                                     const std::vector<Place>& valid_places) {
  auto* main_block = program_desc->MutableBlock(0);
  auto* mul = main_block->AppendOp();
  auto* add = main_block->AppendOp();
  main_block->Var("x");
  main_block->Var("b");
  main_block->Var("mul_out");
  main_block->Var("w");
  main_block->Var("out");

  scope->Var("x")->GetMutable<lite::Tensor>();
  scope->Var("b")->GetMutable<lite::Tensor>();
  scope->Var("mul_out")->GetMutable<lite::Tensor>();
  scope->Var("w")->GetMutable<lite::Tensor>();
  scope->Var("out")->GetMutable<lite::Tensor>();

  mul->SetInput("X", {"x"});
  mul->SetInput("Y", {"w"});
  mul->SetOutput("Out", {"mul_out"});
  mul->SetType("mul");
  mul->SetAttr("x_num_col_dims", 1);
  mul->SetAttr("y_num_col_dims", 1);

  add->SetInput("X", {"mul_out"});
  add->SetInput("Y", {"b"});
  add->SetOutput("Out", {"out"});
  add->SetType("elementwise_add");
  add->SetAttr("axis", 1);

  program_desc->Flush();

  lite::Program program(*program_desc->Proto(), scope, valid_places);
  auto graph = std::unique_ptr<SSAGraph>(new SSAGraph());
  graph->Build(program, valid_places);

  return graph;
}

TEST(pattern_matcher_high_api, graph_test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);

  ASSERT_EQ(graph->nodes().size(), 7UL /*real nodes*/);
  Visualize(graph.get());
}

TEST(pattern_matcher_high_api, fuse_test) {
  framework::ProgramDesc program_desc;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};
  auto scope = std::make_shared<Scope>();
  auto graph = BuildGraph(&program_desc, scope, places);
  const int num_nodes = graph->nodes().size();
  FcFuser fuser;
  fuser(graph.get());
  ASSERT_EQ(graph->nodes().size(),
            num_nodes - 3UL /*nodes removed */ + 1UL /* fused fc node*/);
  Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_LITE_OP(mul);
USE_LITE_OP(elementwise_add);
