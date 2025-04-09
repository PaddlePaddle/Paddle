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

namespace paddle {
namespace framework {
namespace ir {

VarDesc* Data(paddle::framework::BlockDesc* block,
              std::string name,
              std::vector<int64_t> shape = {},
              bool is_persistable = false,
              proto::VarType::Type data_type = proto::VarType::FP32) {
  auto* var = block->Var(name);
  var->SetType(proto::VarType::DENSE_TENSOR);
  var->SetDataType(data_type);
  var->SetShape(shape);
  var->SetPersistable(is_persistable);
  return var;
}

void AddVarToScope(Scope* param_scope,
                   const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<phi::DenseTensor>();
  tensor->Resize(dims);
  auto* cpu_ctx = static_cast<phi::CPUContext*>(
      phi::DeviceContextPool::Instance().Get(phi::CPUPlace()));
  auto* data = cpu_ctx->Alloc<float>(tensor);
  int64_t numel = tensor->numel();
  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 1;
  }
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "matmul0_w", {128, 128});
  return param_scope;
}

int WeightNodeNum(ir::Graph* graph) {
  int num = 0;
  for (auto node : graph->Nodes()) {
    if (node->IsVar() && node->Var()->Persistable()) {
      num++;
    }
  }
  return num;
}

int WeightTensorNum(Scope* scope) {
  int num = 0;
  auto vars = scope->LocalVars();
  for (auto* var : vars) {
    if (var->Get<phi::DenseTensor>().numel() > 0) {
      num++;
    }
  }
  return num;
}

TEST(delete_isolated_node_pass, basic) {
  paddle::framework::ProgramDesc program;
  auto* block0 = program.MutableBlock(0);
  auto* block1 = program.AppendBlock(*block0);

  auto* matmul0_x = Data(block0, "matmul0_x", {1, 128});
  auto* matmul0_w = Data(block0, "matmul0_w", {128, 128}, true);
  auto* matmul0_out = Data(block0, "matmul0_out", {1, 128});
  OpDesc* matmul_op = block0->AppendOp();
  matmul_op->SetType("matmul_v2");
  matmul_op->SetInput("X", {matmul0_x->Name()});
  matmul_op->SetInput("Y", {matmul0_w->Name()});
  matmul_op->SetAttr("trans_x", false);
  matmul_op->SetAttr("trans_y", false);
  matmul_op->SetOutput("Out", {matmul0_out->Name()});

  auto* while_out = Data(block0, "while_out", {1, 128});
  auto* while_step_scopes = Data(block0, "while_step_scopes");
  auto* while_cond = Data(block0, "while_cond");
  OpDesc* while_op = block0->AppendOp();
  while_op->SetType("while");
  while_op->SetInput("X", {matmul0_w->Name(), matmul0_out->Name()});
  while_op->SetInput("Condition", {while_cond->Name()});
  while_op->SetOutput("Out", {while_out->Name()});
  while_op->SetOutput("StepScopes", {while_step_scopes->Name()});
  while_op->SetAttr("sub_block", {block1});
  while_op->SetAttr("is_test", true);

  auto* matmul1_x = Data(block1, matmul0_out->Name(), matmul0_out->GetShape());
  auto* matmul1_w =
      Data(block1, matmul0_w->Name(), matmul0_w->GetShape(), true);
  auto* matmul1_out = Data(block1, "matmul1_out", {1, 128});
  OpDesc* matmul1_op = block1->AppendOp();
  matmul1_op->SetType("matmul_v2");
  matmul1_op->SetInput("X", {matmul1_x->Name()});
  matmul1_op->SetInput("Y", {matmul1_w->Name()});
  matmul1_op->SetAttr("trans_x", false);
  matmul1_op->SetAttr("trans_y", false);
  matmul1_op->SetOutput("Out", {matmul1_out->Name()});

  std::unique_ptr<ir::Graph> graph(new ir::Graph(program));
  auto* scope = CreateParamScope();
  graph->Set("__param_scope__", scope);
  auto pass0 = PassRegistry::Instance().Get("fc_xpu_fuse_pass");
  pass0->Apply(graph.get());
  pass0->Apply(graph->GetSubGraph(1));
  int weight_node_num =
      WeightNodeNum(graph.get()) + WeightNodeNum(graph->GetSubGraph(1));
  PADDLE_ENFORCE_EQ(weight_node_num,
                    6,
                    common::errors::PreconditionNotMet(
                        "Graph should have 6 weight node after "
                        "fc_xpu_fuse_pass, but actually has %d.",
                        weight_node_num));

  auto pass1 = PassRegistry::Instance().Get("delete_isolated_node_pass");
  pass1->Apply(graph.get());
  weight_node_num =
      WeightNodeNum(graph.get()) + WeightNodeNum(graph->GetSubGraph(1));
  PADDLE_ENFORCE_EQ(weight_node_num,
                    4,
                    common::errors::PreconditionNotMet(
                        "Graph should have 4 weight node after "
                        "delete_isolated_node_pass, but actually has %d.",
                        weight_node_num));
  int weight_tensor_num = WeightTensorNum(scope);
  PADDLE_ENFORCE_EQ(weight_tensor_num,
                    2,
                    common::errors::PreconditionNotMet(
                        "Scope should have 2 weight tensor after "
                        "delete_isolated_node_pass, but actually has %d.",
                        weight_tensor_num));

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "while") {
      auto while_in_names = node->Op()->Inputs().at("X");
      PADDLE_ENFORCE_EQ(while_in_names.size(),
                        3,
                        common::errors::PreconditionNotMet(
                            "While op should have 3 input after "
                            "delete_isolated_node_pass, but actually has %d.",
                            while_in_names.size()));
    }
  }

  Scope& scope0 = graph->Get<framework::Scope>("__param_scope__");
  Scope& scope1 =
      graph->GetSubGraph(1)->Get<framework::Scope>("__param_scope__");
  std::vector<std::string> shared_weight_names{matmul0_w->Name() + "_int16",
                                               matmul0_w->Name() + "_max"};
  for (auto name : shared_weight_names) {
    auto* var0 = scope0.FindVar(name);
    auto* var1 = scope1.FindVar(name);
    PADDLE_ENFORCE(
        var0 == var1,
        common::errors::PreconditionNotMet(
            "Variables with the same name in two scopes is different."));
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(delete_isolated_node_pass);
