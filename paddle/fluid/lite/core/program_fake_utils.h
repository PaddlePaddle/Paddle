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

Program FakeProgram() {
  Program program(std::make_shared<lite::Scope>());

  auto add_fc = [&](int id, std::string x) {
    // create variables
    std::string w1 = "w" + std::to_string(id);
    std::string b1 = "b" + std::to_string(id);
    std::string out1 = "out" + std::to_string(id);
    auto w1v = program.scope->Var(w1)->GetMutable<Tensor>();
    auto b1v = program.scope->Var(b1)->GetMutable<Tensor>();
    auto out1v = program.scope->Var(out1)->GetMutable<Tensor>();

    lite::OpDesc desc;
    desc.SetInput("Input", {x});
    desc.SetInput("W", {w1});
    desc.SetInput("Bias", {b1});
    desc.SetOutput("Out", {out1});
    desc.SetType("fc");
    desc.SetAttr<int>("in_num_col_dims", 1);

    // add to input
    program.tmp_vars.push_back(w1);
    program.tmp_vars.push_back(b1);

    auto fc_op = LiteOpRegistry::Global().Create("fc");
    fc_op->Attach(desc, program.scope.get());
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

class ProgramFaker {
 public:
  ProgramFaker() {}

  framework::ProgramDesc* program() {
    desc_.Flush();
    return &desc_;
  }

  void CreateVars(lite::Scope* scope) {
    for (auto& var : tmp_vars_) {
      auto* x = scope->Var(var);
      x->GetMutable<lite::Tensor>();
    }

    for (auto& x : tmp_vars_) {
      desc_.MutableBlock(0)->Var(x);
    }
  }

  void AddMul(const std::string& X, const std::string& Y,
              const std::string& out) {
    tmp_vars_.insert(X);
    tmp_vars_.insert(Y);
    tmp_vars_.insert(out);

    auto* block = desc_.MutableBlock(0);
    auto* op = block->AppendOp();
    op->SetType("mul");
    op->SetInput("X", {X});
    op->SetInput("Y", {Y});
    op->SetOutput("Out", {Y});
    op->SetAttr("x_num_col_dims", 1);
    op->SetAttr("y_num_col_dims", 1);
  }

  void AddFeed(const std::string& Out, int col) {
    tmp_vars_.insert(Out);

    auto* block = desc_.MutableBlock(0);
    auto* op = block->AppendOp();
    op->SetType("feed");
    op->SetInput("X", {"feed"});
    op->SetOutput("Out", {Out});
    op->SetAttr("col", col);
  }

  void AddFetch(const std::string& Input, int col) {
    tmp_vars_.insert(Input);
    auto* block = desc_.MutableBlock(0);
    auto* op = block->AppendOp();
    op->SetType("fetch");
    op->SetInput("X", {Input});
    op->SetOutput("Out", {"fetch"});
    op->SetAttr("col", col);
  }

 private:
  std::set<std::string> tmp_vars_;
  std::vector<std::string> weight_vars_;
  framework::ProgramDesc desc_;
};

}  // namespace lite
}  // namespace paddle
