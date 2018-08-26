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

#include "paddle/fluid/framework/ir/param_opt_pass_base.h"
#include <gtest/gtest.h>
#include <memory>

namespace paddle {
namespace framework {
namespace ir {

class ParamOptTestPass : public ParamOptPassBase {
 protected:
  void RegisterParamOperations(Graph* graph, Scope* scope) const override {
    ToRead("tmp1");
    ToWrite("tmp2");
    ToDrop("tmp1");
    ToCreate("tmp3");
    CheckOrCreateParam(graph, scope);
  }

  // Much operation here.
  void Operate(Graph* graph, Scope* scope) const override {
    ASSERT_TRUE(scope->FindVar("tmp1"));
    ASSERT_TRUE(scope->FindVar("tmp2"));
  }
};

namespace {
void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetInput("Xs", inputs);
  op->SetOutput("Ys", outputs);
}

ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>({"tmp1", "tmp2"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
  }

  SetOp(&prog, "OP0", std::vector<std::string>({"tmp1"}),
        std::vector<std::string>({"tmp2"}));

  return prog;
}

}  // namespace

class ParamOptPassBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    scope_.reset(new Scope);
    auto prog = BuildProgramDesc();
    graph_.reset(new ir::Graph(prog));
    pass_ = PassRegistry::Instance().Get("param_opt_test_pass");
  }

  std::unique_ptr<ir::Graph> graph_;
  std::unique_ptr<ir::Pass> pass_;
  std::unique_ptr<Scope> scope_;
};

TEST_F(ParamOptPassBaseTest, need_param_scope) {
  bool fail = false;
  try {
    pass_->Apply(std::move(graph_));
  } catch (...) {
    // Failed to set param_scope
    fail = true;
  }
  ASSERT_TRUE(fail);
}

TEST_F(ParamOptPassBaseTest, no_var) {
  graph_->Set("param_scope", new Scope*(scope_.get()));

  bool fail = false;
  try {
    pass_->Apply(std::move(graph_));
  } catch (...) {
    // Failed to get param
    fail = true;
  }
  ASSERT_TRUE(fail);
}

TEST_F(ParamOptPassBaseTest, basic) {
  graph_->Set("param_scope", new Scope*(scope_.get()));
  scope_->Var("tmp1");
  scope_->Var("tmp2");

  bool fail = false;
  pass_->Apply(std::move(graph_));
  try {
  } catch (...) {
    // Failed to get param
    fail = true;
  }
  ASSERT_FALSE(fail);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(param_opt_test_pass, paddle::framework::ir::ParamOptTestPass);
