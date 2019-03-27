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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_squash_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn,
           float scale = 0) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);
  if (type == "conv2d") {
    op->SetInput("Input", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2) op->SetInput("Bias", {inputs[2]});
    op->SetOutput("Output", {outputs[0]});
  } else if (type == "quantize") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("Scale", scale);
  } else if (type == "dequantize") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("Scale", scale);
  }
}

// (a,w1,b1)->Conv1->d
// d->Dequant->e
// e->Quant->f
// (f,w2,b2)->Conv2->i
ProgramDesc BuildProgramDesc(bool use_mkldnn, float scale1, float scale2) {
  ProgramDesc prog;
  for (auto& v : std::initializer_list<std::string>(
           {"a", "w1", "b1", "d", "e", "f", "w2", "b2", "i"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("w") == 0 || v.find("b") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "conv2d", "Conv1", {"a", "w1", "b1"}, {"d"}, use_mkldnn);
  SetOp(&prog, "dequantize", "Dequant", {"d"}, {"e"}, use_mkldnn, scale1);
  SetOp(&prog, "quantize", "Quant", {"e"}, {"f"}, use_mkldnn, scale2);
  SetOp(&prog, "conv2d", "Conv2", {"f", "w2", "b2"}, {"i"}, use_mkldnn);
  return prog;
}

static const std::initializer_list<std::string> variable_names{
    "a", "b", "c", "d", "e", "f", "g", "h"};
// a->Conv1->b
// b->Dequant->c
//
// c->Quant1->d and d->Conv2->e
//
// c->Conv3->f
//
// c->Quant2->g and g->Conv4->h
//
ProgramDesc BuildProgramDesc2(bool use_mkldnn, float scale1, float scale2,
                              float scale3) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn);
  SetOp(&prog, "dequantize", "Dequant", {"b"}, {"c"}, use_mkldnn, scale1);

  SetOp(&prog, "quantize", "Quant1", {"c"}, {"d"}, use_mkldnn, scale2);
  SetOp(&prog, "conv2d", "Conv2", {"d"}, {"e"}, use_mkldnn);

  SetOp(&prog, "conv2d", "Conv3", {"c"}, {"f"}, use_mkldnn);

  SetOp(&prog, "quantize", "Quant2", {"c"}, {"g"}, use_mkldnn, scale3);
  SetOp(&prog, "conv2d", "Conv4", {"g"}, {"h"}, use_mkldnn);

  return prog;
}

void InitTensorHolder(Scope* scope, const paddle::platform::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<LoDTensor>();
  tensor->mutable_data(place, proto::VarType::FP32,
                       ::paddle::memory::Allocator::kDefault, 1);
}

void MainTest(const ProgramDesc& prog, int removed_nodes_num) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  // Init scope, as it is used in pass
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  exe.CreateVariables(prog, 0, true, &scope);

  for (auto& v : variable_names) {
    InitTensorHolder(&scope, place, v.c_str());
  }

  graph->Set(kParamScopeAttr, new framework::Scope*(&scope));

  auto pass = PassRegistry::Instance().Get("cpu_quantize_squash_pass");

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(original_nodes_num - removed_nodes_num, current_nodes_num);
}

TEST(CpuQuantizeSquashPass, equal_scales) {
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // Remove 4 nodes: Dequant, Quant, e, f
  auto remove_nodes = 4;
  MainTest(BuildProgramDesc(use_mkldnn, scale, scale), remove_nodes);

  use_mkldnn = !use_mkldnn;
  MainTest(BuildProgramDesc(use_mkldnn, scale, scale), remove_nodes);
}

TEST(CpuQuantizeSquashPass, inequal_scales) {
  auto scale1 = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // Remove 3 nodes: Dequant, Quant, e
  // Insert 1 node: requantize
  auto remove_nodes = 2;
  MainTest(BuildProgramDesc(use_mkldnn, scale1, scale2), remove_nodes);

  use_mkldnn = !use_mkldnn;
  MainTest(BuildProgramDesc(use_mkldnn, scale1, scale2), remove_nodes);
}

TEST(CpuQuantizeSquashPass, branch_to_equal_inequal_and_fp32) {
  // Delete both quantize ops,
  // bypass dequantize in both branches,
  // insert requantize on one branch
  auto scale = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // Remove 3 nodes: Quant1, Quant2, g
  // Insert 1 node: requantize
  auto remove_nodes = 2;
  MainTest(BuildProgramDesc2(use_mkldnn, scale, scale, scale2), remove_nodes);

  use_mkldnn = !use_mkldnn;
  MainTest(BuildProgramDesc2(use_mkldnn, scale, scale, scale2), remove_nodes);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_quantize_squash_pass);
