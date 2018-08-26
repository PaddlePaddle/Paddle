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

#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"
#include "paddle/fluid/framework/lod_tensor.h"

#include <gtest/gtest.h>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  if (type == "lstm") {
    auto* op = prog->MutableBlock(0)->AppendOp();
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Weight", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
    op->SetOutput("BatchCellPreAct", {outputs[0]});
    op->SetOutput("BatchGate", {outputs[1]});
    op->SetOutput("Cell", {outputs[2]});
    op->SetOutput("Hidden", {outputs[3]});
    op->SetType(type);
    std::string candidate_act = "tanh";
    std::string cell_act = "tanh";
    std::string gate_act = "tanh";
    bool is_reverse = false;
    bool use_peepholes = false;
    op->SetAttr("candidate_activation", candidate_act);
    op->SetAttr("cell_activation", cell_act);
    op->SetAttr("gate_activation", gate_act);
    op->SetAttr("is_reverse", is_reverse);
    op->SetAttr("use_peepholes", use_peepholes);
  } else {
    auto* op = prog->MutableBlock(0)->AppendOp();
    op->SetType(type);
    op->SetInput("Xs", inputs);
    op->SetOutput("Ys", outputs);
  }
}

// (a, b)->mul->c
// (c, d)->elementwise_add->e
// (e, f, g)->lstm->(h, i, j, k)
ProgramDesc BuildProgramDesc(Scope* scope) {
  ProgramDesc prog;
  std::unordered_set<std::string> persistable_var = {"b", "d", "f", "g"};
  platform::CPUPlace cpu_place;
  auto input_bias_var = scope->Var("d");
  auto input_bias_tensor = input_bias_var->GetMutable<framework::Tensor>();
  input_bias_tensor->Resize({1, 4 * 10});
  input_bias_tensor->mutable_data<float>(cpu_place);

  auto lstm_bias_var = scope->Var("g");
  auto lstm_bias_tensor = lstm_bias_var->GetMutable<framework::Tensor>();
  lstm_bias_tensor->Resize({1, 4 * 10});
  lstm_bias_tensor->mutable_data<float>(cpu_place);

  for (auto& v : std::vector<std::string>(
           {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
    if (persistable_var.count(v)) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "mul", std::vector<std::string>({"a", "b"}),
        std::vector<std::string>({"c"}));
  SetOp(&prog, "elementwise_add", std::vector<std::string>({"c", "d"}),
        std::vector<std::string>({"e"}));
  SetOp(&prog, "lstm", std::vector<std::string>({"e", "f", "g"}),
        std::vector<std::string>({"h", "i", "j", "k"}));

  return prog;
}

TEST(FCFusePass, basic) {
  std::unique_ptr<Scope> scope(new Scope);

  auto prog = BuildProgramDesc(scope.get());
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  graph->Set("param_scope", new Scope*(scope.get()));
  auto pass = PassRegistry::Instance().Get("fc_lstm_fuse_pass");

  int pre_nodes = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int after_nodes = graph->Nodes().size();

  // Remove 5 Nodes: MUL,ELEMENTWISE_ADD, mul_out
  // elementwise_add_out, LSTM
  // Add 1 Node: FUSION_LSTM
  EXPECT_EQ(pre_nodes - 4, after_nodes);

  // Assert fc op in newly generated graph
  int fc_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "fusion_lstm") {
      ++fc_count;
    }
  }
  EXPECT_EQ(fc_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_lstm_fuse_pass);
