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

#include "paddle/fluid/framework/ir/seqpool_cvm_concat_fuse_pass.h"
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
  } else if (type == "cvm") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("CVM", {inputs[1]});
    op->SetOutput("Y", {outputs[0]});
    op->SetAttr("use_cvm", true);
  } else {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

int CountOpType(const ir::Graph* graph,
                const std::string& op_type = "fusion_seqpool_cvm_concat") {
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
    const std::string& pass_type = "seqpool_cvm_concat_fuse_pass") {
  auto pass = PassRegistry::Instance().Get(pass_type);
  *before = static_cast<int>(graph->Nodes().size());
  graph.reset(pass->Apply(graph.release()));
  *after = static_cast<int>(graph->Nodes().size());
  return graph;
}

/*
 * Before fuse:
 *
 *
 *    a          b          c
 *    |          |          |
 *   op1        op2        op3
 *   / \        / \        / \
 *  d  e  n    f   g   n   h  i   n
 *     |  /        |  /       |  /
 *    op4         op5        op6
 *     |           |          |
       j           k          l
 *     \           |         /
 *               concat
 *                 |
 *                 m
 *
 * Type of op1, op2 and op3 are sequence_pool, with "SUM" pooltype attr.
 * Type of op4, op5 and op6 are cvm, with use_cvm is true.
 *
 * After fuse:
 *    a      b      c      n
 *    \      |      |     /
 *  fusion_seqpool_cvm_concat
 *              |
 *              m
 */
TEST(SeqPoolCVMConcatFusePass, basic) {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a",
                                           "b",
                                           "c",
                                           "d",
                                           "e",
                                           "f",
                                           "g",
                                           "h",
                                           "i",
                                           "j",
                                           "k",
                                           "l",
                                           "m",
                                           "n"})) {
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
        "cvm",
        std::vector<std::string>({"e", "n"}),
        std::vector<std::string>({"j"}));
  SetOp(&prog,
        "cvm",
        std::vector<std::string>({"g", "n"}),
        std::vector<std::string>({"k"}));
  SetOp(&prog,
        "cvm",
        std::vector<std::string>({"i", "n"}),
        std::vector<std::string>({"l"}));
  SetOp(&prog,
        "concat",
        std::vector<std::string>({"j", "k", "l"}),
        std::vector<std::string>({"m"}));

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int before = 0, after = 0;
  graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
  // Remove 16 Nodes: op1, op2, op3, op4, op5, op6, d, e, f, g, h, i, j, k, l,
  // concat_op
  // Add 1 Node: fusion_seqpool_cvm_concat
  EXPECT_EQ(after, before - 15);
  EXPECT_EQ(CountOpType(graph.get()), 1);
}

/*
 * Before fuse:
 *    a               b
 *    |           /       \
 *   op1  k     op2   k   op3
 *   / \ /      / \  /      \
 *  c  d       e   f         g
 *     |           |
 *    op4         op5
 *     |           |
 *     h           i
 *      \         /
 *        concat
 *          |
 *          j
 * Type of op1 and op2 are sequence_pool, with "SUM" pooltype attr.
 * Type of op4 and op5 are cvm, with use_cvm is true.
 *
 * After fuse:
 *   a          k              b
 *    \         |           /     \
 *   fusion_seqpool_cvm_concat    op3
 *              |                  |
 *              j                  g
 */
TEST(SeqPoolCVMConcatFusePass, advanced) {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>(
           {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"})) {
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
        "cvm",
        std::vector<std::string>({"d", "k"}),
        std::vector<std::string>({"h"}));
  SetOp(&prog,
        "cvm",
        std::vector<std::string>({"f", "k"}),
        std::vector<std::string>({"i"}));
  SetOp(&prog,
        "concat",
        std::vector<std::string>({"h", "i"}),
        std::vector<std::string>({"j"}));

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int before = 0, after = 0;
  graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
  // Remove 11 Nodes: op1, op2, op4, op5, c, d, e, f, h, i, concat_op
  // Add 1 Node: fusion_seqpool_cvm_concat
  EXPECT_EQ(after, before - 10);
  EXPECT_EQ(CountOpType(graph.get()), 1);
}

ProgramDesc BuildProgramDesc(int num_inputs_of_concat) {
  ProgramDesc prog;
  auto new_var = [&](const std::string& name) {
    auto* var = prog.MutableBlock(0)->Var(name);
    var->SetType(proto::VarType::DENSE_TENSOR);
  };
  std::vector<std::string> concat_inputs;
  new_var("cvm_in");
  for (int i = 0; i < num_inputs_of_concat; ++i) {
    std::string seqpool_prefix = "seqpool_op_" + std::to_string(i);
    new_var(seqpool_prefix + "in");
    new_var(seqpool_prefix + "out");
    new_var(seqpool_prefix + "out_unused");
    SetOp(&prog,
          "sequence_pool",
          std::vector<std::string>({seqpool_prefix + "in"}),
          std::vector<std::string>(
              {seqpool_prefix + "out_unused", seqpool_prefix + "out"}));

    std::string cvm_prefix = "cvm_op_" + std::to_string(i);
    new_var(cvm_prefix + "out");
    SetOp(&prog,
          "cvm",
          std::vector<std::string>({seqpool_prefix + "out", "cvm_in"}),
          std::vector<std::string>({cvm_prefix + "out"}));

    concat_inputs.push_back(cvm_prefix + "out");
  }
  SetOp(
      &prog, "concat", concat_inputs, std::vector<std::string>({"concat_out"}));
  return prog;
}

// test more inputs of concat
TEST(SeqPoolCVMConcatFusePass, more_inputs) {
  for (int num : {1, 2, 10}) {
    ProgramDesc prog = BuildProgramDesc(num);
    std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
    int before = 0, after = 0;
    graph = GetNumNodesOfBeforeAfter(std::move(graph), &before, &after);
    // Remove Nodes: n * (seqpool_op, seqpool_out, out_unused, cvm_op, cvm_out),
    // and concat_op
    // Add Node: fusion_seqpool_cvm_concat op
    EXPECT_EQ(after, before - num * 5);
    EXPECT_EQ(CountOpType(graph.get()), 1);
  }
}

}  // namespace paddle::framework::ir

USE_PASS(seqpool_cvm_concat_fuse_pass);
