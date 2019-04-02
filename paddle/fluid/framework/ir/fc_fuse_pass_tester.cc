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

#include "paddle/fluid/framework/ir/fc_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "mul") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
    op->SetAttr("x_num_col_dims", {1});
  } else if (type == "elementwise_add") {
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// a->OP0->b
// a->OP1->c
// (b, c)->mul->d
// (d, e)->elementwise_add->f
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a", "b", "c", "d", "e", "f"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "c") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "OP0", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog, "OP1", std::vector<std::string>({"a"}),
        std::vector<std::string>({"c"}));
  SetOp(&prog, "mul", std::vector<std::string>({"b", "c"}),
        std::vector<std::string>({"d"}));
  SetOp(&prog, "elementwise_add", std::vector<std::string>({"d", "e"}),
        std::vector<std::string>({"f"}));

  return prog;
}

TEST(FCFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("fc_fuse_pass");

  int pre_nodes = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));

  int after_nodes = graph->Nodes().size();

  // Remove 3 Nodes: MUL,ELEMENTWISE_ADD, mul_out
  // Add 1 Node: FC
  EXPECT_EQ(pre_nodes - 2, after_nodes);

  // Assert fc op in newly generated graph
  int fc_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fc") {
      ++fc_count;
    }
  }
  EXPECT_EQ(fc_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_fuse_pass);
