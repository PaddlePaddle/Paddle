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

#include "paddle/fluid/framework/ir/seq_concat_fc_fuse_pass.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

void BuildOp(ProgramDesc* prog, const std::string& type,
             const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "mul") {
    CHECK_EQ(inputs.size(), 2);
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
  } else if (type == "sequence_expand") {
    op->SetInput("X", {inputs[0]});
  } else {
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
}

// create a program desc with inner graph include this:
// x0->seq_expand0->\  // NOLINT
// y ----------------> concat->mul->elementwise_add->activation
// x1->seq_expand0->/
ProgramDesc BuildProgramDesc(const std::string& activation_type) {
  ProgramDesc prog;
  std::vector<std::string> vars({"a", "b", "c", "x0", "x1", "y", "expand0_out",
                                 "expand1_out", "concat_out", "weight", "bias",
                                 "mul_out", "elementwise_out", "out"});
  for (auto& v : vars) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weight" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  BuildOp(&prog, "feed_x0", std::vector<std::string>({"a"}),
          std::vector<std::string>({"x0"}));
  BuildOp(&prog, "feed_x1", std::vector<std::string>({"b"}),
          std::vector<std::string>({"x1"}));
  BuildOp(&prog, "feed_y", std::vector<std::string>({"c"}),
          std::vector<std::string>({"y"}));
  // sequence_expand op only need one input here
  BuildOp(&prog, "sequence_expand", std::vector<std::string>({"x0"}),
          std::vector<std::string>({"expand0_out"}));
  BuildOp(&prog, "sequence_expand", std::vector<std::string>({"x1"}),
          std::vector<std::string>({"expand1_out"}));
  BuildOp(&prog, "concat",
          std::vector<std::string>({"expand0_out", "expand1_out", "y"}),
          std::vector<std::string>({"concat_out"}));
  BuildOp(&prog, "mul", std::vector<std::string>({"concat_out", "weight"}),
          std::vector<std::string>({"mul_out"}));
  BuildOp(&prog, "elementwise_add",
          std::vector<std::string>({"mul_out", "bias"}),
          std::vector<std::string>({"elementwise_out"}));
  BuildOp(&prog, activation_type, std::vector<std::string>({"elementwise_out"}),
          std::vector<std::string>({"out"}));
  return prog;
}

TEST(SeqConcatFCPass, basic) {
  auto prog = BuildProgramDesc("relu");
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto* scope = new framework::Scope;
  graph->SetNotOwned(kParamScopeAttr, &scope);
  auto pass = PassRegistry::Instance().Get("seq_concat_fc_fuse_pass");

  int original_nodes_num = graph->Nodes().size();
  graph = pass->Apply(std::move(graph));
  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(original_nodes_num - 10, current_nodes_num);
  int fuse_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fusion_seqexpand_concat_fc") {
      ++fuse_count;
    }
  }
  EXPECT_EQ(fuse_count, 1);
  delete scope;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(seq_concat_fc_fuse_pass);
