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

#include "paddle/fluid/framework/ir/fuse_reshape_transpose_scale_matmul_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "matmul") {
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
  } else if (type == "transpose2") {
    op->SetInput("X", inputs);
    op->SetAttr("axis", std::vector<int>({0, 2, 1, 3}));
  } else if (type == "reshape2") {
    op->SetInput("X", inputs);
    op->SetAttr("shape", std::vector<int>({0, 0, 2, 4}));
  } else if (type == "scale") {
    op->SetInput("X", inputs);
    op->SetAttr("bias", 0.0f);
    op->SetAttr("scale", 1.0f);
  } else {
    op->SetInput("X", inputs);
  }
  op->SetOutput("Out", outputs);
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// Forward:
// a->reshape->b->transpose->c
// (c, e)->matmul->f
// IsScale:
// a->reshape->b->transpose->c->scale->d
//  (d, e)->matmul->f
// Backward:
// (a, b)->matmul->c->transpose->d->reshape->e
ProgramDesc BuildProgramDesc(bool is_scale, bool is_out, bool is_x) {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a", "b", "c", "d", "e", "f"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
  }
  if (is_out) {
    SetOp(&prog, "matmul", std::vector<std::string>({"a", "b"}),
          std::vector<std::string>({"c"}));
    SetOp(&prog, "transpose2", std::vector<std::string>({"c"}),
          std::vector<std::string>({"d"}));
    SetOp(&prog, "reshape2", std::vector<std::string>({"d"}),
          std::vector<std::string>({"e"}));
  } else {
    SetOp(&prog, "reshape2", std::vector<std::string>({"a"}),
          std::vector<std::string>({"b"}));
    SetOp(&prog, "transpose2", std::vector<std::string>({"b"}),
          std::vector<std::string>({"c"}));
    if (is_scale) {
      SetOp(&prog, "scale", std::vector<std::string>({"c"}),
            std::vector<std::string>({"d"}));
      if (is_x) {
        SetOp(&prog, "matmul", std::vector<std::string>({"d", "e"}),
              std::vector<std::string>({"f"}));
      } else {
        SetOp(&prog, "matmul", std::vector<std::string>({"e", "d"}),
              std::vector<std::string>({"f"}));
      }
    } else {
      if (is_x) {
        SetOp(&prog, "matmul", std::vector<std::string>({"c", "e"}),
              std::vector<std::string>({"f"}));
      } else {
        SetOp(&prog, "matmul", std::vector<std::string>({"e", "c"}),
              std::vector<std::string>({"f"}));
      }
    }
  }

  return prog;
}

void MainTest(bool is_scale = false, bool is_out = false, bool is_x = true) {
  auto prog = BuildProgramDesc(is_scale, is_out, is_x);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  // Init scope, as it is used in pass
  exe.CreateVariables(prog, 0, true, &scope);
  graph->Set(kParamScopeAttr, new framework::Scope*(&scope));

  auto pass =
      PassRegistry::Instance().Get("fuse_reshape_transpose_scale_matmul_pass");

  graph.reset(pass->Apply(graph.release()));

  // Assert fused matmul op in newly generated graph
  int fused_matmul_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() &&
        node->Op()->Type() == "fused_matmul_reshape_transpose") {
      ++fused_matmul_count;
    }
  }
  EXPECT_EQ(fused_matmul_count, 1);
}

TEST(ReshapeTransposeScaleMatmulFusePass, reshape_transpose_scale_matmul_x) {
  MainTest(true, false, true);
}
TEST(ReshapeTransposeScaleMatmulFusePass, reshape_transpose_scale_matmul_y) {
  MainTest(true, false, false);
}
TEST(ReshapeTransposeScaleMatmulFusePass, reshape_transpose_matmul_x) {
  MainTest(false, false, true);
}
TEST(ReshapeTransposeScaleMatmulFusePass, reshape_transpose_matmul_y) {
  MainTest(false, false, false);
}
TEST(ReshapeTransposeScaleMatmulFusePass, matmul_transpose_reshape) {
  MainTest(false, true, false);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fuse_reshape_transpose_scale_matmul_pass);
