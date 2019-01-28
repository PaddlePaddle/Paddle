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

#include "paddle/fluid/framework/details/graph_print_pass.h"
#include "paddle/fluid/framework/details/graph_test_base.h"

REGISTER_OPERATOR(sum, paddle::framework::DummyOp,
                  paddle::framework::SumOpMaker);
REGISTER_OPERATOR(split, paddle::framework::DummyOp,
                  paddle::framework::SplitOpMaker);
REGISTER_OPERATOR(assign, paddle::framework::DummyOp,
                  paddle::framework::AssignOpMaker,
                  paddle::framework::DummyVarTypeInference);

/*
  a @ b
    c
  d @ e
 */

using paddle::framework::ProgramDesc;
using paddle::framework::proto::VarType;

inline static ProgramDesc FillProgramDesc() {
  ProgramDesc prog;
  prog.MutableBlock(0)->Var("a")->SetType(VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("b")->SetType(VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("c")->SetType(VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("d")->SetType(VarType::LOD_TENSOR);
  prog.MutableBlock(0)->Var("e")->SetType(VarType::LOD_TENSOR);
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("sum");
    op->SetInput("X", {"a", "b"});
    op->SetOutput("Out", {"c"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("split");
    op->SetInput("X", {"c"});
    op->SetOutput("Out", {"d", "e"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("sum");
    op->SetInput("X", {"d", "e"});
    op->SetOutput("Out", {"d"});
  }
  {
    auto* op = prog.MutableBlock(0)->AppendOp();
    op->SetType("assign");
    op->SetInput("X", {"d"});
    op->SetOutput("Out", {"d"});
  }
  return prog;
}

namespace paddle {
namespace framework {
namespace details {

TEST(SSAGraphPrinter, Normal) {
  auto program = FillProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  graph->Set<GraphvizNodes>(kGraphviz, new GraphvizNodes);
  std::unique_ptr<SSAGraphPrinter> printer(new SSAGraphPrinterImpl);

  // redirect debug graph to a file.
  constexpr char graph_path[] = "graph_print_pass.txt";
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_path));
  PADDLE_ENFORCE(fout->good());
  printer->Print(*graph, *fout);
}

using ir::Graph;
using ir::Node;
void BuildCircleGraph(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);

  o1->outputs.push_back(v1);
  o1->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o1);
}

void BuildCircleGraph2(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);

  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);

  o2->outputs.push_back(v2);
  o1->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o1);
}

void BuildNoCircleGraph(Graph* g) {
  ir::Node* o1 = g->CreateEmptyNode("op1", Node::Type::kOperation);
  ir::Node* o2 = g->CreateEmptyNode("op2", Node::Type::kOperation);
  ir::Node* o3 = g->CreateEmptyNode("op3", Node::Type::kOperation);
  ir::Node* o4 = g->CreateEmptyNode("op4", Node::Type::kOperation);
  ir::Node* o5 = g->CreateEmptyNode("op5", Node::Type::kOperation);
  ir::Node* v1 = g->CreateEmptyNode("var1", Node::Type::kVariable);
  ir::Node* v2 = g->CreateEmptyNode("var2", Node::Type::kVariable);
  ir::Node* v3 = g->CreateEmptyNode("var3", Node::Type::kVariable);
  ir::Node* v4 = g->CreateEmptyNode("var4", Node::Type::kVariable);

  // o1->v1->o2
  o1->outputs.push_back(v1);
  o2->inputs.push_back(v1);
  v1->inputs.push_back(o1);
  v1->outputs.push_back(o2);
  // o2->v2->o3
  // o2->v2->o4
  o2->outputs.push_back(v2);
  o3->inputs.push_back(v2);
  o4->inputs.push_back(v2);
  v2->inputs.push_back(o2);
  v2->outputs.push_back(o3);
  v2->outputs.push_back(o4);
  // o2->v3->o5
  o2->outputs.push_back(v3);
  o5->inputs.push_back(v3);
  v3->inputs.push_back(o2);
  v3->outputs.push_back(o5);
  // o3-v4->o5
  o3->outputs.push_back(v4);
  o5->inputs.push_back(v4);
  v4->inputs.push_back(o3);
  v4->outputs.push_back(o5);

  // o2->v3->o1
  v3->outputs.push_back(o1);
  o1->inputs.push_back(v3);
}

TEST(SSAGraphPrinter, SimpleCircle) {
  ProgramDesc prog;

  Graph graph(prog);
  BuildCircleGraph(&graph);
  ASSERT_TRUE(HasCircle(graph));

  graph.Set<GraphvizNodes>(kGraphviz, new GraphvizNodes);
  std::unique_ptr<SSAGraphPrinter> printer(new SSAGraphPrinterImpl);

  // redirect debug graph to a file.
  constexpr char graph_path[] = "graph_print_pass_simple_circle.txt";
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_path));
  PADDLE_ENFORCE(fout->good());
  printer->Print(graph, *fout);
}

TEST(SSAGraphPrinter, ComplexCircle) {
  ProgramDesc prog;
  Graph graph(prog);
  BuildCircleGraph2(&graph);
  ASSERT_TRUE(HasCircle(graph));

  graph.Set<GraphvizNodes>(kGraphviz, new GraphvizNodes);
  std::unique_ptr<SSAGraphPrinter> printer(new SSAGraphPrinterImpl);

  // redirect debug graph to a file.
  constexpr char graph_path[] = "graph_print_pass_complex_circle.txt";
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_path));
  PADDLE_ENFORCE(fout->good());
  printer->Print(graph, *fout);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
