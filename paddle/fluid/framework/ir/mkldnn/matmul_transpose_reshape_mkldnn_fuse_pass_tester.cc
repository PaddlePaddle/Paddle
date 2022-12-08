// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/matmul_transpose_reshape_mkldnn_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc *prog,
           const std::string &type,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs) {
  auto *op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("X", {inputs[0]});
  op->SetOutput("Out", {outputs[0]});
  if (type == "transpose2") {
    op->SetAttr("axis", std::vector<int>({0, 2, 1, 3}));
    op->SetOutput("XShape", {outputs[1]});
  }
  if (type == "reshape2") {
    op->SetAttr("shape", std::vector<int>({4, 5, 6}));
    op->SetOutput("XShape", {outputs[1]});
  }

  if (type == "matmul") {
    op->SetInput("Y", {inputs[1]});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("alpha", 1.0f);
    op->SetAttr("transpose_X", true);
    op->SetAttr("transpose_Y", true);
  }
  if (type == "matmul_v2") {
    op->SetInput("Y", {inputs[1]});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("trans_x", true);
    op->SetAttr("trans_y", true);
  }
}

ProgramDesc BuildProgramDesc(const std::string &op_name) {
  ProgramDesc prog;
  for (auto &v : std::initializer_list<std::string>(
           {"a1", "a2", "b", "c", "cx", "d", "dx", "e"})) {
    auto *var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
  }

  SetOp(&prog, op_name, {"a1", "a2"}, {"b"});
  SetOp(&prog, "transpose2", {"b"}, {"c", "cx"});
  SetOp(&prog, "reshape2", {"c"}, {"d", "dx"});
  SetOp(&prog, "fc", {"d"}, {"e"});

  return prog;
}

void MainTest(const ProgramDesc &prog, const std::string &op_name) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  int original_nodes_num = graph->Nodes().size();

  auto pass =
      PassRegistry::Instance().Get("matmul_transpose_reshape_mkldnn_fuse_pass");
  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();
  EXPECT_EQ(original_nodes_num - 6, current_nodes_num);

  for (auto *node : graph->Nodes()) {
    if (node->IsOp()) {
      auto *op = node->Op();
      if (op->Type() == op_name) {
        EXPECT_EQ(op->GetAttrIfExists<std::vector<int>>("fused_reshape_Out"),
                  std::vector<int>({4, 5, 6}));
        EXPECT_EQ(op->GetAttrIfExists<std::vector<int>>("fused_transpose_Out"),
                  std::vector<int>({0, 2, 1, 3}));
      }
    }
  }
}

TEST(MatmulTransposeReshapeFusePass, matmul_fuse_pass) {
  auto prog = BuildProgramDesc("matmul");
  MainTest(prog, "matmul");
}

TEST(MatmulTransposeReshapeFusePass, matmul_v2_fuse_pass) {
  auto prog = BuildProgramDesc("matmul_v2");
  MainTest(prog, "matmul_v2");
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(matmul_transpose_reshape_mkldnn_fuse_pass);
