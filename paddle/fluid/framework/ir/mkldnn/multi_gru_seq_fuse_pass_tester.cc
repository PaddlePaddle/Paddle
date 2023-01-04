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

#include <initializer_list>

#include "paddle/fluid/framework/ir/mkldnn/multi_gru_seq_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

const std::vector<std::string> churn_out_vars(ProgramDesc* prog,
                                              const std::string& prefix,
                                              int number) {
  auto v = std::vector<std::string>();
  for (int i = 0; i < number; ++i) {
    auto name = prefix + std::to_string(i);
    prog->MutableBlock(0)->Var(name);
    v.push_back(name);
  }
  return v;
}

void create_vars(ProgramDesc* prog,
                 const std::initializer_list<std::string>& names) {
  for (auto name : names) prog->MutableBlock(0)->Var(name);
}

void SetMultiGruOp(ProgramDesc* prog,
                   const std::string x,
                   const std::vector<std::string> wx,
                   const std::vector<std::string> wh,
                   const std::vector<std::string> b,
                   const std::string h,
                   int layers,
                   bool origin_mode) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType("multi_gru");
  op->SetInput("X", {x});
  op->SetInput("WeightX", wx);
  op->SetInput("WeightH", wh);
  op->SetInput("Bias", b);
  op->SetOutput("Hidden", {h});
  op->SetAttr("layers", layers);
  op->SetAttr("origin_mode", origin_mode);
}

// (x, wx1, wh1, b1) -> multi_gru1 -> h1
// (h1, wx2, wh2, b2) -> multi_gru2 -> h2
void MainTest(int layers1, int layers2, bool origin_mode1, bool origin_mode2) {
  ProgramDesc prog;

  // Create variables
  create_vars(&prog, {"x", "h1", "h2"});
  const std::vector<std::string> wx1 =
      churn_out_vars(&prog, "wx1", 2 * layers1);
  const std::vector<std::string> wx2 =
      churn_out_vars(&prog, "wx2", 2 * layers2);
  const std::vector<std::string> wh1 =
      churn_out_vars(&prog, "wh1", 2 * layers1);
  const std::vector<std::string> wh2 =
      churn_out_vars(&prog, "wh2", 2 * layers2);
  const std::vector<std::string> b1 = churn_out_vars(&prog, "b1", 2 * layers1);
  const std::vector<std::string> b2 = churn_out_vars(&prog, "b2", 2 * layers2);

  // Create program descriptor
  SetMultiGruOp(&prog, "x", wx1, wh1, b1, "h1", layers1, origin_mode1);
  SetMultiGruOp(&prog, "h1", wx2, wh2, b2, "h2", layers2, origin_mode2);

  // Apply pass
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  Scope scope;
  graph->SetNotOwned(kParamScopeAttr, &scope);
  int original_nodes_num = graph->Nodes().size();
  auto pass = PassRegistry::Instance().Get("multi_gru_seq_fuse_pass");
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  // Verify graph after fuse
  bool should_fuse = origin_mode1 == origin_mode2;
  int count_multi_gru = 0;
  auto layers = layers1;
  auto wx = wx1;
  auto wh = wh1;
  auto b = b1;
  auto h = "h1";
  if (should_fuse) {
    layers += layers2;
    wx.insert(wx.end(), wx2.begin(), wx2.end());
    wh.insert(wh.end(), wh2.begin(), wh2.end());
    b.insert(b.end(), b2.begin(), b2.end());
    h = "h2";
  }
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "multi_gru") {
        if (op->Input("X")[0] == "x") {
          EXPECT_EQ(op->GetAttrIfExists<int>("layers"), layers);
          EXPECT_EQ(op->Input("WeightX").size(), 2u * layers);
          EXPECT_EQ(op->Input("WeightH").size(), 2u * layers);
          EXPECT_EQ(op->Input("Bias").size(), 2u * layers);
          for (int i = 0; i < 2 * layers; ++i) {
            EXPECT_EQ(op->Input("WeightX")[i], wx[i]);
            EXPECT_EQ(op->Input("WeightH")[i], wh[i]);
            EXPECT_EQ(op->Input("Bias")[i], b[i]);
          }
          EXPECT_EQ(op->Output("Hidden")[0], h);
          EXPECT_EQ(op->GetAttrIfExists<bool>("origin_mode"), origin_mode1);
        } else {
          EXPECT_EQ(op->GetAttrIfExists<int>("layers"), layers2);
          EXPECT_EQ(op->Input("X")[0], "h1");
          EXPECT_EQ(op->Input("WeightX").size(), 2u * layers2);
          EXPECT_EQ(op->Input("WeightH").size(), 2u * layers2);
          EXPECT_EQ(op->Input("Bias").size(), 2u * layers2);
          for (int i = 0; i < 2 * layers2; ++i) {
            EXPECT_EQ(op->Input("WeightX")[i], wx2[i]);
            EXPECT_EQ(op->Input("WeightH")[i], wh2[i]);
            EXPECT_EQ(op->Input("Bias")[i], b2[i]);
          }
          EXPECT_EQ(op->Output("Hidden")[0], "h2");
          EXPECT_EQ(op->GetAttrIfExists<bool>("origin_mode"), origin_mode2);
        }
        ++count_multi_gru;
      }
    }
  }

  // If the fuse is applied, then:
  // nodes to be removed: 2x multi_gru + 1x hidden(output)
  // nodes to be added: multi_gru
  // If the fuse is not applied, then:
  // nodes to be removed: none
  // nodes to be added: none
  const int removed_nodes_count = should_fuse ? 3 : 0;
  const int added_nodes_count = should_fuse ? 1 : 0;

  EXPECT_EQ(original_nodes_num - removed_nodes_count + added_nodes_count,
            current_nodes_num);
  EXPECT_EQ(count_multi_gru, should_fuse ? 1 : 2);
}

TEST(MultiGruSeqFusePass, same_origin_modes_1) {
  int layers1 = 1;
  int layers2 = 1;
  bool origin_mode1 = false;
  bool origin_mode2 = false;
  MainTest(layers1, layers2, origin_mode1, origin_mode2);
}

TEST(MultiGruSeqFusePass, same_origin_modes_2) {
  int layers1 = 2;
  int layers2 = 3;
  bool origin_mode1 = false;
  bool origin_mode2 = false;
  MainTest(layers1, layers2, origin_mode1, origin_mode2);
}

TEST(MultiGruSeqFusePass, same_origin_modes_3) {
  int layers1 = 2;
  int layers2 = 1;
  bool origin_mode1 = true;
  bool origin_mode2 = true;
  MainTest(layers1, layers2, origin_mode1, origin_mode2);
}

TEST(MultiGruSeqFusePass, different_origin_modes) {
  int layers1 = 2;
  int layers2 = 2;
  bool origin_mode1 = true;
  bool origin_mode2 = false;
  MainTest(layers1, layers2, origin_mode1, origin_mode2);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(multi_gru_seq_fuse_pass);
