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

#include "paddle/fluid/framework/ir/mkldnn/scale_matmul_fuse_pass.h"
#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, float scale = 1.0f,
           float bias = 0.0f) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);
  if (type == "scale") {
    op->SetInput("X", {inputs[0]});
    op->SetAttr("scale", scale);
    op->SetAttr("bias", bias);
  } else if (type == "matmul") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    op->SetAttr("alpha", scale);
  } else {
    FAIL() << "Unexpected operator type.";
  }
  op->SetOutput("Out", {outputs[0]});
}

// a->scale->b
// (b,c)->matmul->d
ProgramDesc BuildProgramDesc(float scale, float bias, float alpha) {
  ProgramDesc prog;

  for (auto& v : std::vector<std::string>({"a", "b", "c", "d"})) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "scale", {"a"}, {"b"}, scale, bias);
  SetOp(&prog, "matmul", {"b", "c"}, {"d"}, alpha);
  return prog;
}

void MainTest(const ProgramDesc& prog, int removed_nodes_count,
              const std::vector<std::string> scale_in_out,
              const std::vector<std::string> matmul_in_out, float alpha) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num = graph->Nodes().size();
  auto pass = PassRegistry::Instance().Get("scale_matmul_fuse_pass");
  graph.reset(pass->Apply(graph.release()));
  int current_nodes_num = graph->Nodes().size();

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "scale") {
        EXPECT_EQ(op->Input("X")[0], scale_in_out[0]);
        EXPECT_EQ(op->Output("Out")[0], scale_in_out[1]);
      } else if (op->Type() == "matmul") {
        EXPECT_EQ(op->Input("X")[0], matmul_in_out[0]);
        EXPECT_EQ(op->Input("Y")[0], matmul_in_out[1]);
        EXPECT_EQ(op->Output("Out")[0], matmul_in_out[2]);
        EXPECT_EQ(op->GetAttrIfExists<float>("alpha"), alpha);
      }
    }
  }
  EXPECT_EQ(original_nodes_num - removed_nodes_count, current_nodes_num);
}

TEST(ScaleMatmulFusePass, scale_matmul_with_no_bias) {
  auto bias = 0.0f;
  auto scale = 2.34f;
  auto alpha = 3.45f;
  int removed_nodes_count = 2;
  MainTest(BuildProgramDesc(scale, bias, alpha), removed_nodes_count, {},
           {"a", "c", "d"}, scale * alpha);
}

TEST(ScaleMatmulFusePass, scale_matmul_with_bias) {
  auto bias = 1.0f;
  auto scale = 2.34f;
  auto alpha = 3.45f;
  int removed_nodes_count = 0;
  MainTest(BuildProgramDesc(scale, bias, alpha), removed_nodes_count,
           {"a", "b"}, {"b", "c", "d"}, alpha);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(scale_matmul_fuse_pass);
