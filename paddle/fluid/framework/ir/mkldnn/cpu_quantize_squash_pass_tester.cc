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
           float scale = 0, float bias = 0.0) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);
  if (type == "conv2d") {
    op->SetAttr("Scale_out", scale);
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
  } else if (type == "requantize") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("Scale_out", scale);
  } else if (type == "concat") {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
  } else if (type == "fc") {
    op->SetInput("Input", {inputs[0]});
    PADDLE_ENFORCE_EQ(inputs.size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The fc inputs should contain input and weights, but "
                          "now the size of inputs is %d",
                          inputs.size()));
    op->SetInput("W", {inputs[1]});
    op->SetOutput("Out", outputs);
  } else if (type == "scale") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("scale", scale);
    op->SetAttr("bias", bias);
  }
}

// (a,w1,b1)->Conv1->d
// d->Dequant(scale1)->e
// e->Quant(scale2)->f
// (f,w2,b2)->Conv2->i
ProgramDesc BuildConvRequantProgramDesc(bool use_mkldnn, float scale_out,
                                        float scale1, float scale2) {
  ProgramDesc prog;
  for (auto& v : std::initializer_list<std::string>(
           {"a", "w1", "b1", "d", "e", "f", "w2", "b2", "i"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("w") == 0 || v.find("b") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "conv2d", "Conv1", {"a", "w1", "b1"}, {"d"}, use_mkldnn,
        scale_out);
  SetOp(&prog, "dequantize", "Dequant", {"d"}, {"e"}, use_mkldnn, scale1);
  SetOp(&prog, "quantize", "Quant", {"e"}, {"f"}, use_mkldnn, scale2);
  SetOp(&prog, "conv2d", "Conv2", {"f", "w2", "b2"}, {"i"}, use_mkldnn,
        scale_out);
  return prog;
}

static const std::initializer_list<std::string> variable_names{
    "a", "b", "c", "d", "e", "f", "g", "h"};

// a->Conv1->b
// b->Dequant(scale1)->c
// c->Quant1(scale2)->d and d->Conv2->e
// c->Conv3->f
// c->Quant2(scale3)->g and g->Conv4->h
ProgramDesc BuildConvMultiOutputProgramDesc(bool use_mkldnn, float scale_out,
                                            float scale1, float scale2,
                                            float scale3) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "dequantize", "Dequant", {"b"}, {"c"}, use_mkldnn, scale1);

  SetOp(&prog, "quantize", "Quant1", {"c"}, {"d"}, use_mkldnn, scale2);
  SetOp(&prog, "conv2d", "Conv2", {"d"}, {"e"}, use_mkldnn, scale_out);

  SetOp(&prog, "conv2d", "Conv3", {"c"}, {"f"}, use_mkldnn, scale_out);

  SetOp(&prog, "quantize", "Quant2", {"c"}, {"g"}, use_mkldnn, scale3);
  SetOp(&prog, "conv2d", "Conv4", {"g"}, {"h"}, use_mkldnn, scale_out);

  return prog;
}

//  a->Conv1->b->Requant(scale1)->c
//  d->Conv2->e->Requant(scale2)->f
//  {c,f}->Concat
ProgramDesc BuildConvsRequantConcatProgramDesc(bool use_mkldnn, float scale_out,
                                               float scale1, float scale2) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "requantize", "Requant1", {"b"}, {"c"}, use_mkldnn, scale1);

  SetOp(&prog, "conv2d", "Conv2", {"d"}, {"e"}, use_mkldnn, scale_out);
  SetOp(&prog, "requantize", "Requant2", {"e"}, {"f"}, use_mkldnn, scale2);

  SetOp(&prog, "concat", "Concat", {"c"}, {"f"}, use_mkldnn);

  return prog;
}

// a->Concat->b
// b->Dequant(scale1)->c
// c->Quant(scale2)->d
// d->Conv->e
ProgramDesc BuildConcatDequantQuantProgramDesc(bool use_mkldnn, float scale_out,
                                               float scale1, float scale2) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "concat", "Concat", {"a"}, {"b"}, use_mkldnn);
  SetOp(&prog, "dequantize", "Dequant", {"b"}, {"c"}, use_mkldnn, scale1);
  SetOp(&prog, "quantize", "Quant", {"c"}, {"d"}, use_mkldnn, scale2);
  SetOp(&prog, "conv2d", "Conv2", {"d"}, {"e"}, use_mkldnn, scale_out);
  return prog;
}

// a->Conv1->b
// b->Requant1(Scale1)->c
// b->Requant2(Scale2)->d
ProgramDesc BuildConvMultiRequantProgramDesc(bool use_mkldnn, float scale_out,
                                             float scale1, float scale2) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "requantize", "Requant1", {"b"}, {"c"}, use_mkldnn, scale1);
  SetOp(&prog, "requantize", "Requant2", {"b"}, {"d"}, use_mkldnn, scale2);
  return prog;
}

// a->Conv1->b
// b->Dequant1(Scale1)->c
// c->Concat
ProgramDesc BuildConvDequantConcatProgramDesc(bool use_mkldnn, float scale_out,
                                              float scale) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "dequantize", "Dequant1", {"b"}, {"c"}, use_mkldnn, scale);
  SetOp(&prog, "concat", "Concat1", {"c"}, {"d"}, use_mkldnn);
  return prog;
}

// a->fc->b
// b->Dequant1->c
// c->Concat1->d
ProgramDesc BuildFcDequantConcatProgramDesc(bool use_mkldnn, float scale_out,
                                            float scale) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "fc", "Fc1", {"a", "w1"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "dequantize", "Dequant1", {"b"}, {"c"}, use_mkldnn, scale);
  SetOp(&prog, "concat", "Concat1", {"c"}, {"d"}, use_mkldnn);
  return prog;
}

// a->fc->b
// b->Dequant1->c
// b->concat->d
ProgramDesc BuildFcDequantFcProgramDesc(bool use_mkldnn, float scale_out,
                                        float scale) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "fc", "Fc1", {"a", "w1"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "dequantize", "Dequant1", {"b"}, {"c"}, use_mkldnn, scale);
  SetOp(&prog, "concat", "Concat1", {"b"}, {"d"}, use_mkldnn);
  return prog;
}

// a->Conv1->b
// b->Dequant1(Scale1)->c
// b->Conv2->d
ProgramDesc BuildConvDequantConvProgramDesc(bool use_mkldnn, float scale_out,
                                            float scale) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, scale_out);
  SetOp(&prog, "dequantize", "Dequant1", {"b"}, {"c"}, use_mkldnn, scale);
  SetOp(&prog, "conv2d", "Conv2", {"b"}, {"d"}, use_mkldnn);
  return prog;
}

// a->concat->b
// b->Quant1(Scale1)->c
// b->Quant2(Scale2)->d
// b->concat->e
// c->fc->f
// d->fc->g
ProgramDesc BuildMultipleQuantizeProgramDesc(bool use_mkldnn, float first_scale,
                                             float second_scale) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "concat", "Concat1", {"a"}, {"b"}, use_mkldnn);
  SetOp(&prog, "quantize", "Quantize1", {"b"}, {"c"}, use_mkldnn, first_scale);
  SetOp(&prog, "quantize", "Quantize2", {"b"}, {"d"}, use_mkldnn, second_scale);
  SetOp(&prog, "concat", "Concat2", {"b"}, {"e"}, use_mkldnn);
  SetOp(&prog, "fc", "Fc1", {"c", "w1"}, {"f"}, use_mkldnn, first_scale);
  SetOp(&prog, "fc", "Fc2", {"d", "w2"}, {"g"}, use_mkldnn, second_scale);

  return prog;
}

// a->Dequant->b
// b->Scale->c
ProgramDesc BuildDequantScaleProgramDesc(bool use_mkldnn, float dequant_scale,
                                         float scale_scale, float bias) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequant", {"a"}, {"b"}, use_mkldnn,
        dequant_scale);
  SetOp(&prog, "scale", "Scale", {"b"}, {"c"}, use_mkldnn, scale_scale, bias);

  return prog;
}

void InitTensorHolder(Scope* scope, const paddle::platform::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<LoDTensor>();
  tensor->mutable_data(place, proto::VarType::FP32, 1);
}

void PrepareGraph(std::unique_ptr<ir::Graph>* graph, const ProgramDesc& prog) {
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  exe.CreateVariables(prog, 0, true, &scope);

  for (auto& v : variable_names) {
    InitTensorHolder(&scope, place, v.c_str());
  }
  (*graph)->SetNotOwned(kParamScopeAttr, &scope);
}

void RegisterPass(std::unique_ptr<ir::Graph>* graph) {
  auto pass = PassRegistry::Instance().Get("cpu_quantize_squash_pass");
  graph->reset(pass->Apply(graph->release()));
}

// check number of nodes
void CountNodeTest(const ProgramDesc& prog, int removed_nodes_num) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  PrepareGraph(&graph, prog);

  int original_nodes_num = graph->Nodes().size();
  RegisterPass(&graph);
  int current_nodes_num = graph->Nodes().size();

  EXPECT_EQ(original_nodes_num - removed_nodes_num, current_nodes_num);
}

// check op->scale_out
void EqualScaleTest(const ProgramDesc& prog, const std::string& op_name,
                    const std::string& scale_name, float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  PrepareGraph(&graph, prog);
  RegisterPass(&graph);

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() &&
        boost::get<std::string>(node->Op()->GetAttr("name")) == op_name) {
      float op_scale = boost::get<float>(node->Op()->GetAttr(scale_name));
      EXPECT_EQ(op_scale, scale);
    }
  }
}

// check requant_op scales
void CheckRequantScalesTest(const ProgramDesc& prog, float scale_in,
                            float scale_out) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  PrepareGraph(&graph, prog);
  RegisterPass(&graph);

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "requantize") {
      float op_scale_in = boost::get<float>(node->Op()->GetAttr("Scale_in"));
      EXPECT_EQ(op_scale_in, scale_in);
      float op_scale_out = boost::get<float>(node->Op()->GetAttr("Scale_out"));
      EXPECT_EQ(op_scale_out, scale_out);
    }
  }
}

// check requant_op scales
void IsForceFp32OutputTest(const ProgramDesc& prog, std::string op_type,
                           bool target_is_force_fp32_output) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  PrepareGraph(&graph, prog);
  RegisterPass(&graph);

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == op_type) {
      bool is_force_fp32_output =
          node->Op()->GetAttrIfExists<bool>("force_fp32_output");
      EXPECT_EQ(is_force_fp32_output, target_is_force_fp32_output);
    }
  }
}

// From Conv1->d->Dequant->e->Quant->f->Conv2
// To Conv1->d->Conv2
TEST(CpuQuantizeSquashPass, equal_scales) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // Remove 4 nodes: Dequant, Quant, e, f
  auto remove_nodes = 4;

  CountNodeTest(
      BuildConvRequantProgramDesc(use_mkldnn, scale_out, scale, scale),
      remove_nodes);
}

// From Conv1->d->Dequant->e->Quant->f->Conv2
// First change to Conv1->d->Requant->f->Conv2
// Then Conv1->f->Conv2
TEST(CpuQuantizeSquashPass, unequal_scales) {
  auto scale_out = 1.0f;
  auto scale1 = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // Remove 4 nodes: Dequant, Quant, e, d
  auto remove_nodes = 4;

  CountNodeTest(
      BuildConvRequantProgramDesc(use_mkldnn, scale_out, scale1, scale2),
      remove_nodes);

  EqualScaleTest(
      BuildConvRequantProgramDesc(use_mkldnn, scale_out, scale1, scale2),
      "Conv1", "Scale_out", scale2);
}

//  a->Conv1->b->Requant->c
//  d->Conv2->e->Requant->f
//  {c,f}->Concat
TEST(CpuQuantizeSquashPass, equal_scales_squash_requantize) {
  // Delete both requantize op
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // Remove 4 nodes: b, Requant1, e, Requant2
  auto remove_nodes = 4;
  CountNodeTest(
      BuildConvsRequantConcatProgramDesc(use_mkldnn, scale_out, scale, scale),
      remove_nodes);

  // check equal scale conv->scale_out and requant->scale_out
  EqualScaleTest(
      BuildConvsRequantConcatProgramDesc(use_mkldnn, scale_out, scale, scale),
      "Conv1", "Scale_out", scale);
  EqualScaleTest(
      BuildConvsRequantConcatProgramDesc(use_mkldnn, scale_out, scale, scale),
      "Conv2", "Scale_out", scale);
}

// from
// a->Conv1->b->Dequant(Scale1)->c
// c->Quant1(Scale1)->d and d->Conv2->e
// c->Quant2(Scale2)->g and g->Conv4->h
// c->Conv3->f
// to
// a->Conv1->b
// b->Conv2->e
// b->Requant(Scale_in = Scale1; Scale_out = Scale2)->g->Conv4->h
// b->Dequant(Scale1)->c->Conv3->f
TEST(CpuQuantizeSquashPass, branch_to_equal_unequal_and_fp32) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // Remove 3 nodes: Quant1, c, Quant2,
  // Insert 1 node: Requant
  auto remove_nodes = 2;

  CountNodeTest(BuildConvMultiOutputProgramDesc(use_mkldnn, scale_out, scale,
                                                scale, scale2),
                remove_nodes);
  CheckRequantScalesTest(BuildConvMultiOutputProgramDesc(use_mkldnn, scale_out,
                                                         scale, scale, scale2),
                         scale, scale2);
}

// a->Concat->b->Dequant->c->Quant->d->Conv->e
// to a->Concat->b->Requant->d->Conv->e
TEST(CpuQuantizeSquashPass,
     unequal_scales_squash_dequantize_quantize_into_requantize) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // Remove 3 nodes: Dequant1, c, Quant
  // Insert 1 node: Requant
  auto remove_nodes = 2;

  CountNodeTest(
      BuildConcatDequantQuantProgramDesc(use_mkldnn, scale_out, scale, scale2),
      remove_nodes);
  CheckRequantScalesTest(
      BuildConcatDequantQuantProgramDesc(use_mkldnn, scale_out, scale, scale2),
      scale, scale2);
}

// a->Conv1->b
// b->Requant1(Scale1)->c
// b->Requant2(Scale2)->d
TEST(CpuQuantizeSquashPass, more_than_one_conv_out_outputs) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto scale2 = 21.0f;
  auto use_mkldnn = true;
  // nothing change
  auto remove_nodes = 0;
  CountNodeTest(
      BuildConvMultiRequantProgramDesc(use_mkldnn, scale_out, scale, scale2),
      remove_nodes);
}

// a->Conv1->c->Concat
TEST(CpuQuantizeSquashPass, conv_dequant_only_one_output) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // remove 2 nodes: Dequant1, c
  auto remove_nodes = 2;
  CountNodeTest(BuildConvDequantConcatProgramDesc(use_mkldnn, scale_out, scale),
                remove_nodes);
  IsForceFp32OutputTest(
      BuildConvDequantConcatProgramDesc(use_mkldnn, scale_out, scale), "conv2d",
      true);
}

// If there are more than one op after conv->dequantize, do not fuse
TEST(CpuQuantizeSquashPass, conv_dequant_more_than_one_op_after_conv) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // nothing change
  auto remove_nodes = 0;
  CountNodeTest(BuildConvDequantConvProgramDesc(use_mkldnn, scale_out, scale),
                remove_nodes);
  IsForceFp32OutputTest(
      BuildConvDequantConvProgramDesc(use_mkldnn, scale_out, scale), "conv2d",
      false);
}

// from
// a->fc->b->Dequant1->c->Concat1->d
// to
// a->fc->c->Concat->d
TEST(CpuQuantizeSquashPass, fc_dequant_only_one_output) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // remove 2 nodes: b, Dequant1
  auto remove_nodes = 2;
  CountNodeTest(BuildFcDequantConcatProgramDesc(use_mkldnn, scale_out, scale),
                remove_nodes);
  IsForceFp32OutputTest(
      BuildFcDequantConcatProgramDesc(use_mkldnn, scale_out, scale), "fc",
      true);
}

// If there are more than one op after fc->dequantize, do not fuse
TEST(CpuQuantizeSquashPass, fc_dequant_more_than_one_op_after_dequant) {
  auto scale_out = 1.0f;
  auto scale = 1.2345f;
  auto use_mkldnn = true;
  // nothing change
  auto remove_nodes = 0;
  CountNodeTest(BuildFcDequantFcProgramDesc(use_mkldnn, scale_out, scale),
                remove_nodes);
  IsForceFp32OutputTest(
      BuildFcDequantFcProgramDesc(use_mkldnn, scale_out, scale), "fc", false);
}

// a->Concat1->b
// b->Concat2
// b->Quatize1(Scale)->c
// c->Fc1
// c->Fc2
TEST(CpuQuantizeSquashPass, quatize_with_same_scale) {
  auto first_scale = 1.2345f;
  auto second_scale = 1.2345f;
  auto use_mkldnn = true;
  // remove nodes: Quantize2 + d
  auto remove_nodes = 1 + 1;
  CountNodeTest(
      BuildMultipleQuantizeProgramDesc(use_mkldnn, first_scale, second_scale),
      remove_nodes);
}

// if scales are not the same, do not fuse
TEST(CpuQuantizeSquashPass, quatize_with_different_scale) {
  auto first_scale = 1.2345f;
  auto second_scale = 1.5432f;
  auto use_mkldnn = true;
  // nothing change
  auto remove_nodes = 0;
  CountNodeTest(
      BuildMultipleQuantizeProgramDesc(use_mkldnn, first_scale, second_scale),
      remove_nodes);
}

// if scale has no bias
TEST(CpuQuantizeSquashPass, dequantize_scale_with_no_bias) {
  auto dequant_scale = 1.2345f;
  auto scale_scale = 1.5432f;
  auto bias = 0.0f;
  auto use_mkldnn = true;
  // remove: dequant out, scale op
  auto remove_nodes = 2;
  CountNodeTest(BuildDequantScaleProgramDesc(use_mkldnn, dequant_scale,
                                             scale_scale, bias),
                remove_nodes);
  EqualScaleTest(BuildDequantScaleProgramDesc(use_mkldnn, dequant_scale,
                                              scale_scale, bias),
                 "Dequant", "Scale", dequant_scale / scale_scale);
}

// if scale has bias
TEST(CpuQuantizeSquashPass, dequantize_scale_with_bias) {
  auto dequant_scale = 1.2345f;
  auto scale_scale = 1.5432f;
  auto bias = 1.0f;
  auto use_mkldnn = true;
  // nothing change
  auto remove_nodes = 0;
  CountNodeTest(BuildDequantScaleProgramDesc(use_mkldnn, dequant_scale,
                                             scale_scale, bias),
                remove_nodes);
  EqualScaleTest(BuildDequantScaleProgramDesc(use_mkldnn, dequant_scale,
                                              scale_scale, bias),
                 "Dequant", "Scale", dequant_scale);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_quantize_squash_pass);
