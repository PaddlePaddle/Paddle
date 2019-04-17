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

#pragma once
#include <string>
#include "paddle/fluid/lite/core/mir/ssa_graph.h"
#include "paddle/fluid/lite/core/op_registry.h"

namespace paddle {
namespace lite {

mir::Program FakeProgram() {
  mir::Program program;
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

}  // namespace lite
}  // namespace paddle
