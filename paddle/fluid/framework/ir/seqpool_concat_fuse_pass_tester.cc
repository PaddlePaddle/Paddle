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

#include "paddle/fluid/framework/ir/seqpool_concat_fuse_pass.h"
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
  if (type == "sequence_pool") {
    op->SetInput("X", {inputs[0]});
    std::string pooltype = "SUM";
    op->SetAttr("pooltype", pooltype);
    op->SetOutput("MaxIndex", {outputs[0]});
    op->SetOutput("Out", {outputs[1]});
  } else if (type == "concat") {
    op->SetInput("X", inputs);
    op->SetAttr("axis", 1);
    op->SetOutput("Out", {outputs[0]});
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

/*
 * Before fuse:
 *    a         b         c
 *    |         |         |
 *   op1       op2       op3
 *   / \       / \       / \
 *  d  e      f   g     h   i
 *      \         |        /
 *            concat
 *              |
 *              j
 * After fuse:
 *    a         b         c
 *    \         |        /
 *    fusion_seqpool_concat
 *              |
 *              j
 * unused nodes: d, f, h
 */
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>(
           {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
  }

  SetOp(&prog, "sequence_pool", std::vector<std::string>({"a"}),
        std::vector<std::string>({"d", "e"}));
  SetOp(&prog, "sequence_pool", std::vector<std::string>({"b"}),
        std::vector<std::string>({"f", "g"}));
  SetOp(&prog, "sequence_pool", std::vector<std::string>({"c"}),
        std::vector<std::string>({"h", "i"}));
  SetOp(&prog, "concat", std::vector<std::string>({"e", "g", "i"}),
        std::vector<std::string>({"j"}));

  return prog;
}

TEST(SeqPoolConcatFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("seqpool_concat_fuse_pass");

  int pre_nodes = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int after_nodes = graph->Nodes().size();

  // Remove 7 Nodes: op1, op2, op3, e, g, i, concat_op
  // Add 1 Node: fusion_seqpool_concat
  EXPECT_EQ(pre_nodes - 6, after_nodes);

  // Assert new op in newly generated graph
  int count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fusion_seqpool_concat") {
      ++count;
    }
  }
  EXPECT_EQ(count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(seqpool_concat_fuse_pass);
