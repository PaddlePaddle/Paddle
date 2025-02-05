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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/seqpool_concat_fuse_pass.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle::framework::ir {

void SetOp(ProgramDesc* prog,
           const std::string& type,
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
  } else {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

int CountOpType(const ir::Graph* graph,
                const std::string& op_type = "fusion_seqpool_concat") {
  int count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == op_type) {
      ++count;
    }
  }
  return count;
}

std::unique_ptr<ir::Graph> GetNumNodesOfBeforeAfter(
    std::unique_ptr<ir::Graph> graph,
    int* before,
    int* after,
    const std::string& pass_type = "seqpool_concat_fuse_pass") {
  auto pass = PassRegistry::Instance().Get(pass_type);
  *before = static_cast<int>(graph->Nodes().size());
  graph.reset(pass->Apply(graph.release()));
  *after = static_cast<int>(graph->Nodes().size());
  return graph;
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
 * Type of op1, op2 and op3 are sequence_pool, with "SUM" pooltype attr
 *
 * After fuse:
 *    a         b         c
 *    \         |        /
 *    fusion_seqpool_concat
 *              |
 *              j
 */
TEST(SeqPoolConcatFusePass, basic) {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>(
           {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::DENSE_TENSOR);
  }

  SetOp(&prog,
        "sequence_pool",
        std::vector<std::string>({"a"}),
        std::vector<std::string>({"d", "e"}));
  SetOp(&prog,
        "sequence_pool",
        std::vector<std::string>({"b"}),
        std::vector<std::string>({"f", "g"}));
  SetOp(&prog,
        "sequence_pool",
        std::vector<std::string>({"c"}),
        std::vector<std::string>({"h", "i"}));
  SetOp(&prog,
        "concat",
        std::vector<std::string>({"e", "g", "i"}),
        std::vector<std::string>({"j"}));

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int before = 0, after = 0;
  graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
  // Remove 10 Nodes: op1, op2, op3, d, e, f, g, h, i, concat_op
  // Add 1 Node: fusion_seqpool_concat
  EXPECT_EQ(after, before - 9);
  EXPECT_EQ(CountOpType(graph.get()), 1);
}

/*
 * Before fuse:
 *    a            b
 *    |           /  \
 *   op1        op2  op3
 *   / \        / \    \
 *  c  d       e   f    g
 *      \         /
 *        concat
 *          |
 *          h
 * Type of op1 and op2 are sequence_pool, with "SUM" pooltype attr
 *
 * After fuse:
 *   a                         b
 *    \                     /     \
 *    fusion_seqpool_concat       op3
 *              |                  |
 *              h                  g
 */
TEST(SeqPoolConcatFusePass, advanced) {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "d", "e", "f", "g", "h"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::DENSE_TENSOR);
  }

  SetOp(&prog,
        "sequence_pool",
        std::vector<std::string>({"a"}),
        std::vector<std::string>({"c", "d"}));
  SetOp(&prog,
        "sequence_pool",
        std::vector<std::string>({"b"}),
        std::vector<std::string>({"e", "f"}));
  SetOp(&prog,
        "op3",
        std::vector<std::string>({"b"}),
        std::vector<std::string>({"g"}));
  SetOp(&prog,
        "concat",
        std::vector<std::string>({"d", "f"}),
        std::vector<std::string>({"h"}));

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int before = 0, after = 0;
  graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
  // Remove 7 Nodes: op1, op2, c, d, e, f concat_op
  // Add 1 Node: fusion_seqpool_concat
  EXPECT_EQ(after, before - 6);
  EXPECT_EQ(CountOpType(graph.get()), 1);
}

ProgramDesc BuildProgramDesc(int num_inputs_of_concat) {
  ProgramDesc prog;
  auto new_var = [&](const std::string& name) {
    auto* var = prog.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::DENSE_TENSOR);
  };
  std::vector<std::string> concat_inputs;
  for (int i = 0; i < num_inputs_of_concat; ++i) {
    std::string prefix = "seqpool_op_" + std::to_string(i);
    new_var(prefix + "in");
    new_var(prefix + "out");
    new_var(prefix + "out_unused");
    SetOp(&prog,
          "sequence_pool",
          std::vector<std::string>({prefix + "in"}),
          std::vector<std::string>({prefix + "out", prefix + "out_unused"}));
    concat_inputs.push_back(prefix + "out");
  }
  SetOp(
      &prog, "concat", concat_inputs, std::vector<std::string>({"concat_out"}));
  return prog;
}

// test more inputs of concat
TEST(SeqPoolConcatFusePass, more_inputs) {
  for (int num : {1, 2, 10}) {
    ProgramDesc prog = BuildProgramDesc(num);
    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    int before = 0, after = 0;
    graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
    // Remove Nodes: n * (seqpool_op, out, out_unused), and concat_op
    // Add Node: fusion_seqpool_concat op
    EXPECT_EQ(after, before - num * 3);
    EXPECT_EQ(CountOpType(graph.get()), 1);
  }
}

}  // namespace paddle::framework::ir

USE_PASS(seqpool_concat_fuse_pass);
