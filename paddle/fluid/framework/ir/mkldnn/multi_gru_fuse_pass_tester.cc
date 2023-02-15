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

#include "paddle/fluid/framework/ir/mkldnn/multi_gru_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           bool is_reverse = false,
           bool origin_mode = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);
  if (type == "fusion_gru") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("WeightX", {inputs[1]});
    op->SetInput("WeightH", {inputs[2]});
    op->SetInput("Bias", {inputs[3]});
    op->SetOutput("Hidden", {outputs[0]});
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("origin_mode", origin_mode);
  } else if (type == "concat") {
    op->SetInput("X", {inputs[0], inputs[1]});
    op->SetOutput("Out", {outputs[0]});
  } else {
    FAIL() << "Unexpected operator type.";
  }
}

static const std::initializer_list<std::string> variable_names = {
    "x", "wx1", "wx2", "wh1", "wh2", "b1", "b2", "h1", "h2", "out"};

// (x, wx1, wh1, b1) -> fusion_gru1 -> h1
// (x, wx2, wh2, b2) -> fusion_gru2 -> h2
// (h1, h2) -> concat -> out
ProgramDesc BuildProgramDesc(bool origin_mode1, bool origin_mode2) {
  ProgramDesc prog;

  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog,
        "fusion_gru",
        {"x", "wx1", "wh1", "b1"},
        {"h1"},
        false,
        origin_mode1);
  SetOp(&prog,
        "fusion_gru",
        {"x", "wx2", "wh2", "b2"},
        {"h2"},
        true,
        origin_mode2);
  SetOp(&prog, "concat", {"h1", "h2"}, {"out"});
  return prog;
}

void MainTest(const ProgramDesc& prog,
              int removed_nodes_count,
              int added_nodes_count,
              const std::vector<std::string> multi_gru_inputs,
              const std::string multi_gru_output,
              bool origin_mode) {
  // Apply pass
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  Scope scope;
  graph->SetNotOwned(kParamScopeAttr, &scope);
  int original_nodes_num = graph->Nodes().size();
  auto pass = PassRegistry::Instance().Get("multi_gru_fuse_pass");
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  // Verify graph after fuse
  int count_multi_gru = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "multi_gru") {
        EXPECT_EQ(op->Input("X")[0], multi_gru_inputs[0]);
        EXPECT_EQ(op->Input("WeightX").size(), 2u);
        EXPECT_EQ(op->Input("WeightX")[0], multi_gru_inputs[1]);
        EXPECT_EQ(op->Input("WeightX")[1], multi_gru_inputs[2]);
        EXPECT_EQ(op->Input("WeightH").size(), 2u);
        EXPECT_EQ(op->Input("WeightH")[0], multi_gru_inputs[3]);
        EXPECT_EQ(op->Input("WeightH")[1], multi_gru_inputs[4]);
        EXPECT_EQ(op->Input("Bias").size(), 2u);
        EXPECT_EQ(op->Input("Bias")[0], multi_gru_inputs[5]);
        EXPECT_EQ(op->Input("Bias")[1], multi_gru_inputs[6]);
        EXPECT_EQ(op->Output("Hidden")[0], multi_gru_output);
        EXPECT_EQ(op->GetAttrIfExists<int>("layers"), 1);
        EXPECT_EQ(op->GetAttrIfExists<bool>("origin_mode"), origin_mode);
        ++count_multi_gru;
      }
    }
  }
  EXPECT_EQ(original_nodes_num - removed_nodes_count + added_nodes_count,
            current_nodes_num);
  EXPECT_EQ(count_multi_gru, added_nodes_count);
}

TEST(MultiGruFusePass, same_origin_modes_1) {
  bool origin_mode1 = false;
  bool origin_mode2 = false;

  // nodes to be removed: 2x fusion_gru + 2x hidden(output) + concat
  const int removed_nodes_count = 5;
  // nodes to be added: multi_gru
  const int added_nodes_count = 1;

  const std::initializer_list<std::string> multi_gru_inputs = {
      "x", "wx1", "wx2", "wh1", "wh2", "b1", "b2"};
  MainTest(BuildProgramDesc(origin_mode1, origin_mode2),
           removed_nodes_count,
           added_nodes_count,
           multi_gru_inputs,
           "out",
           origin_mode1);
}

TEST(MultiGruFusePass, same_origin_modes_2) {
  bool origin_mode1 = true;
  bool origin_mode2 = true;

  // nodes to be removed: 2x fusion_gru + 2x hidden(output) + concat
  const int removed_nodes_count = 5;
  // nodes to be added: multi_gru
  const int added_nodes_count = 1;

  const std::initializer_list<std::string> multi_gru_inputs = {
      "x", "wx1", "wx2", "wh1", "wh2", "b1", "b2"};
  MainTest(BuildProgramDesc(origin_mode1, origin_mode2),
           removed_nodes_count,
           added_nodes_count,
           multi_gru_inputs,
           "out",
           origin_mode1);
}

TEST(MultiGruFusePass, different_origin_modes) {
  bool origin_mode1 = true;
  bool origin_mode2 = false;

  // the fuse should not be applied, so
  // nodes to be removed: none
  const int removed_nodes_count = 0;
  // nodes to be added: none
  const int added_nodes_count = 0;

  const std::initializer_list<std::string> multi_gru_inputs = {
      "x", "wx1", "wx2", "wh1", "wh2", "b1", "b2"};
  MainTest(BuildProgramDesc(origin_mode1, origin_mode2),
           removed_nodes_count,
           added_nodes_count,
           multi_gru_inputs,
           "out",
           origin_mode1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(multi_gru_fuse_pass);
