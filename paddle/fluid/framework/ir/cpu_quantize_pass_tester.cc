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

#include "paddle/fluid/framework/ir/cpu_quantize_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn,
           bool use_quantizer = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);
  if (type == "conv2d") {
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2)
      op->SetInput("Bias", {inputs[2]});
    else
      op->SetInput("Bias", {});
    if (inputs.size() > 3) {
      op->SetInput("ResidualData", {inputs[3]});
      op->SetAttr("fuse_residual_connection", true);
    } else {
      op->SetInput("ResidualData", {});
      op->SetAttr("fuse_residual_connection", false);
    }
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("use_quantizer", use_quantizer);
    op->SetAttr("Scale_in", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
    op->SetAttr("Scale_weights", std::vector<float>{1.0f});
  } else if (type == "pool2d") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("use_quantizer", use_quantizer);
  } else if (type == "dropout") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
  } else if (type == "fc") {
    op->SetInput("Input", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("W", {inputs[1]});
    if (inputs.size() > 2) op->SetInput("Bias", {inputs[2]});
    op->SetOutput("Out", {outputs[0]});
  }
}

static const std::initializer_list<std::string> variable_names{
    "a", "w1", "c",  "d", "w2", "e",  "f", "g",
    "h", "w3", "b1", "i", "j",  "w4", "b2"};
// (a,w1)->Conv1->c and c->Pool1->d
//
// (d,w2)->Conv2->e and e->Pool2->f
//
// d->Dropout1->g and g->Fc1->h and (h,w3,b1,i)->Conv3->j
//
// (d,w4, b2)->Conv4->i
ProgramDesc BuildProgramDesc(bool use_mkldnn, bool use_quantizer) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("w") == 0 || v.find("b") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "conv2d", "Conv1", {"a", "w1"}, {"c"}, use_mkldnn,
        use_quantizer);
  SetOp(&prog, "pool2d", "Pool1", {"c"}, {"d"}, use_mkldnn, use_quantizer);

  SetOp(&prog, "conv2d", "Conv2", {"d", "w2"}, {"e"}, use_mkldnn,
        use_quantizer);
  SetOp(&prog, "pool2d", "Pool2", {"e"}, {"f"}, use_mkldnn, use_quantizer);

  SetOp(&prog, "dropout", "Dropout1", {"d"}, {"g"}, use_mkldnn);
  SetOp(&prog, "fc", "Fc1", {"g"}, {"h"}, use_mkldnn);
  SetOp(&prog, "conv2d", "Conv3", {"h", "w3", "b1", "i"}, {"j"}, use_mkldnn,
        use_quantizer);

  SetOp(&prog, "conv2d", "Conv4", {"c", "w4", "b2"}, {"i"}, use_mkldnn,
        use_quantizer);

  return prog;
}

void InitTensorHolder(Scope* scope, const paddle::platform::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<LoDTensor>();
  tensor->mutable_data(place, proto::VarType::FP32,
                       ::paddle::memory::Allocator::kDefault, 1);
}

void MainTest(const ProgramDesc& prog, int conv_count, int pool_count,
              int quant_count, int dequant_count, int added_nodes_count,
              float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  // Init scope, as it is used in pass
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  exe.CreateVariables(prog, 0, true, &scope);

  auto* scales = new VarQuantScale();

  for (auto& v : variable_names) {
    InitTensorHolder(&scope, place, v.c_str());
    LoDTensor tensor;
    tensor.Resize({1});
    auto* ptr = tensor.mutable_data<double>(place);
    ptr[0] = 2.0;

    (*scales)[v] = std::make_pair(false, std::move(tensor));
  }

  graph->Set(kParamScopeAttr, new framework::Scope*(&scope));

  auto pass = PassRegistry::Instance().Get("cpu_quantize_pass");
  pass->Set("quant_var_scales", scales);

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int conv2d_nodes_count = 0;
  int pool2d_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "conv2d") {
        conv2d_nodes_count++;
        auto op_name = boost::get<std::string>(op->GetAttr("name"));
        EXPECT_EQ(boost::get<float>(op->GetAttr("Scale_in")), scale)
            << "Scale_in for node '" + op_name + "'.";
        EXPECT_EQ(boost::get<float>(op->GetAttr("Scale_out")), scale)
            << "Scale_out for node '" + op_name + "'.";
        EXPECT_EQ(
            boost::get<std::vector<float>>(op->GetAttr("Scale_weights"))[0],
            scale)
            << "Scale_weights for node '" + op_name + "'.";
      } else if (op->Type() == "pool2d") {
        pool2d_nodes_count++;
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }
  EXPECT_EQ(conv2d_nodes_count, conv_count);
  EXPECT_EQ(pool2d_nodes_count, pool_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, quantize) {
  bool use_mkldnn = true;
  bool use_quantizer = true;
  // (a->QUANT1->IN1,w1)->Conv1->OUT1->DEQUANT1->c and
  // c->QUANT2->IN2->Pool1->OUT2->DEQUANT2->d
  //
  // (d->QUANT3->IN3,w2)->Conv2->OUT3->DEQUANT3->e and
  // e->QUANT4->IN4->Pool2->OUT4->DEQUANT4->f
  //
  // d->Dropout1->g and g->Fc1->h and
  // (h->QUANT5->IN5,w3,b1,i->QUANT6->IN6)->Conv3->OUT5->DEQUANT5->j
  //
  // (d->QUANT7->IN7,w4, b2)->Conv4->DEQUANT6->OUT6->i
  // Insert nodes: 7 Quant + 7 IN + 6 OUT + 6 DEQUANT
  int added_nodes = 7 + 7 + 6 + 6;
  MainTest(BuildProgramDesc(use_mkldnn, use_quantizer), 4, 2, 7, 6, added_nodes,
           2.0f * 127);
}

TEST(CpuQuantizePass, do_not_quantize) {
  bool use_mkldnn = true;
  bool use_quantizer = false;
  int added_nodes = 0;
  MainTest(BuildProgramDesc(use_mkldnn, use_quantizer), 4, 2, 0, 0, added_nodes,
           1.0f);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_quantize_pass);
