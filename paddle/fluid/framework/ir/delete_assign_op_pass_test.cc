// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle::framework::ir {

TEST(delete_assign_op_pass, basic) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("assign_x");
  auto* out_var = program.MutableBlock(0)->Var("assign_out");
  out_var->SetName(x_var->Name());
  OpDesc* assign_op = program.MutableBlock(0)->AppendOp();
  assign_op->SetType("assign");
  assign_op->SetInput("X", {x_var->Name()});
  assign_op->SetOutput("Out", {out_var->Name()});

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("delete_assign_op_pass");
  graph.reset(pass->Apply(graph.release()));
  int assign_num = GetNumOpNodes(graph, "assign");
  PADDLE_ENFORCE_EQ(
      assign_num,
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 assign after delete_assign_op_pass, "
          "but actually has %d.",
          assign_num));
}

}  // namespace paddle::framework::ir

USE_PASS(delete_assign_op_pass);
