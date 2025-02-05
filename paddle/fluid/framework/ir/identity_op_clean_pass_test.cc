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

TEST(identity_op_clean_pass, assign) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("assign_x");
  auto* out_var = program.MutableBlock(0)->Var("assign_out");
  out_var->SetName(x_var->Name());
  OpDesc* assign_op = program.MutableBlock(0)->AppendOp();
  assign_op->SetType("assign");
  assign_op->SetInput("X", {x_var->Name()});
  assign_op->SetOutput("Out", {out_var->Name()});

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("identity_op_clean_pass");
  graph.reset(pass->Apply(graph.release()));
  int assign_num = GetNumOpNodes(graph, "assign");
  PADDLE_ENFORCE_EQ(
      assign_num,
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 assign after identity_op_clean_pass, "
          "but actually has %d.",
          assign_num));
}

TEST(identity_op_clean_pass, scale) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("scale_x");
  auto* out_var = program.MutableBlock(0)->Var("scale_out");
  OpDesc* scale_op = program.MutableBlock(0)->AppendOp();
  scale_op->SetType("scale");
  scale_op->SetInput("X", {x_var->Name()});
  scale_op->SetOutput("Out", {out_var->Name()});
  scale_op->SetAttr("scale", 1.f);
  scale_op->SetAttr("bias", 0.f);

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("identity_op_clean_pass");
  graph.reset(pass->Apply(graph.release()));
  int scale_num = GetNumOpNodes(graph, "scale");
  PADDLE_ENFORCE_EQ(
      scale_num,
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 scale op after identity_op_clean_pass, "
          "but actually has %d.",
          scale_num));
}

TEST(identity_op_clean_pass, cast) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("cast_x");
  auto* out_var = program.MutableBlock(0)->Var("cast_out");
  OpDesc* cast_op = program.MutableBlock(0)->AppendOp();
  cast_op->SetType("cast");
  cast_op->SetInput("X", {x_var->Name()});
  cast_op->SetOutput("Out", {out_var->Name()});
  cast_op->SetAttr("in_dtype", 5);
  cast_op->SetAttr("out_dtype", 5);

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("identity_op_clean_pass");
  graph.reset(pass->Apply(graph.release()));
  int cast_num = GetNumOpNodes(graph, "cast");
  PADDLE_ENFORCE_EQ(
      cast_num,
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 cast after identity_op_clean_pass, "
          "but actually has %d.",
          cast_num));
}

TEST(identity_op_clean_pass, concat) {
  ProgramDesc program;
  auto* x_var = program.MutableBlock(0)->Var("concat_x");
  auto* out_var = program.MutableBlock(0)->Var("concat_out");
  OpDesc* concat_op = program.MutableBlock(0)->AppendOp();
  concat_op->SetType("concat");
  concat_op->SetInput("X", {x_var->Name()});
  concat_op->SetOutput("Out", {out_var->Name()});

  std::unique_ptr<Graph> graph(new Graph(program));
  auto pass = PassRegistry::Instance().Get("identity_op_clean_pass");
  graph.reset(pass->Apply(graph.release()));
  int concat_num = GetNumOpNodes(graph, "concat");
  PADDLE_ENFORCE_EQ(
      concat_num,
      0,
      common::errors::PreconditionNotMet(
          "graph should have 0 concat after identity_op_clean_pass, "
          "but actually has %d.",
          concat_num));
}

}  // namespace paddle::framework::ir

USE_PASS(identity_op_clean_pass);
