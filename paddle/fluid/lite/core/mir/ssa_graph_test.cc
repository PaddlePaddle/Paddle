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

#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/lite/core/mir/graph_visualize_pass.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void BuildFc(framework::ProgramDesc* desc, const std::string& x,
             const std::string& w, const std::string& b,
             const std::string& out) {
  auto* fc = desc->MutableBlock(0)->AppendOp();
  fc->SetInput("Input", {x});
  fc->SetInput("W", {w});
  fc->SetInput("Bias", {b});
  fc->SetOutput("Out", {out});
}

Program FakeProgram() {
  Program program;
  program.scope = new lite::Scope;

  auto add_fc = [&](int id, std::string x) {
    // create variables
    std::string w1 = "w" + std::to_string(id);
    std::string b1 = "b" + std::to_string(id);
    std::string out1 = "out" + std::to_string(id);
    auto w1v = program.scope->Var(w1)->GetMutable<Tensor>();
    auto b1v = program.scope->Var(b1)->GetMutable<Tensor>();
    auto out1v = program.scope->Var(out1)->GetMutable<Tensor>();

    framework::OpDesc desc;
    desc.SetInput("Input", {x});
    desc.SetInput("W", {w1});
    desc.SetInput("Bias", {b1});
    desc.SetOutput("Out", {out1});
    desc.SetType("fc");
    desc.SetAttr("in_num_col_dims", 1);
    desc.Flush();

    // add to input
    program.tmp_vars.push_back(w1);
    program.tmp_vars.push_back(b1);

    auto fc_op = LiteOpRegistry::Global().Create("fc");
    fc_op->PickKernel({Place{TARGET(kHost), PRECISION(kFloat)}});
    fc_op->Attach(desc, program.scope);
    program.ops.emplace_back(std::move(fc_op));

    w1v->Resize({100, 100});
    b1v->Resize({100, 1});
    out1v->Resize({100, 100});

    return out1;
  };

  // x1, w1, b1 -fc-> out1
  // out1, w2, b2 -fc-> out2

  std::string x = "x";
  program.tmp_vars.push_back(x);
  auto* xv = program.scope->Var(x)->GetMutable<Tensor>();
  xv->Resize({100, 100});

  for (int i = 0; i < 3; i++) {
    x = add_fc(i, x);
  }
  return program;
}

TEST(SSAGraph, test) {
  auto program = FakeProgram();
  SSAGraph graph;
  std::vector<Place> places{{TARGET(kHost), PRECISION(kFloat)}};

  graph.Build(program, places);

  Visualize(&graph);
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(fc);
USE_LITE_KERNEL(fc, kHost, kFloat);
