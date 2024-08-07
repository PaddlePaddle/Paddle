// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

void SetOp(ProgramDesc* prog,
           const std::string& type,
           const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("name", name);
  op->SetInput("X", inputs);
  op->SetOutput("Out", outputs);
}

// (a, conv_w)->conv2d->b
// (b, bn_scale, bn_bias, mean, var)->batch_norm
//     ->(c, mean, var, save_mean, save_inv_var)
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"a",
                                           "conv_w",
                                           "b",
                                           "bn_scale",
                                           "bn_bias",
                                           "mean",
                                           "var",
                                           "c",
                                           "save_mean",
                                           "save_inv_var"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v == "conv_w" || v == "bn_scale" || v == "bn_bias" || v == "mean" ||
        v == "var") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog,
        "conv2d",
        "conv",
        std::vector<std::string>({"a", "conv_w"}),
        std::vector<std::string>({"b"}));
  SetOp(&prog,
        "batch_norm",
        "bn",
        std::vector<std::string>({"b", "bn_scale", "bn_bias", "mean", "var"}),
        std::vector<std::string>(
            {"c", "mean", "var", "save_mean", "save_inv_var"}));
  return prog;
}

TEST(IsTestPass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("sync_batch_norm_pass");

  graph.reset(pass->Apply(graph.release()));

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      auto op_name = PADDLE_GET_CONST(std::string, op->GetAttr("name"));
      if (op_name == "bn") {
        ASSERT_EQ(op->Type(), "sync_batch_norm");
      }
    }
  }
}

}  // namespace paddle::framework::ir

USE_PASS(sync_batch_norm_pass);
