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

#include "paddle/fluid/framework/ir/mkldnn_correct_test_phase_pass.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn = false,
           bool is_test = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("X", inputs);
  op->SetOutput("Out", outputs);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("is_test", is_test);
}

// a->pool2d->b
// b->relu->c
// c->pool2d->d
// d->hard_sigmoid->e
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;

  SetOp(&prog, "pool2d", "pooling1", std::vector<std::string>({"a"}),
        std::vector<std::string>({"b"}), true, true);
  SetOp(&prog, "relu", "Relu", std::vector<std::string>({"b"}),
        std::vector<std::string>({"c"}), true, false);
  SetOp(&prog, "pool2d", "pooling2", std::vector<std::string>({"c"}),
        std::vector<std::string>({"d"}), false, false);
  SetOp(&prog, "hard_sigmoid", "HardSigmoid", std::vector<std::string>({"d"}),
        std::vector<std::string>({"e"}), false, true);

  return prog;
}

TEST(ConvReLUFusePass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("mkldnn_correct_test_phase_pass");

  graph = pass->Apply(std::move(graph));

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("is_test"));
      EXPECT_TRUE(boost::get<bool>(op->GetAttr("is_test")));
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(mkldnn_correct_test_phase_pass);
