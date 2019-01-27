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

}  // namespace details
}  // namespace framework
}  // namespace paddle
