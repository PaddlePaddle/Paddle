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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/imperative/type_defs.h"
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
  } else if (type == "pool2d" || type == "transpose2" || type == "reshape2") {
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
    op->SetAttr("use_quantizer", use_quantizer);
    op->SetAttr("Scale_in", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
    op->SetAttr("Scale_weights", std::vector<float>{1.0f});
  } else if (type == "concat") {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
    op->SetAttr("use_quantizer", use_quantizer);
  } else if (type == "dequantize") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("Scale", 1.0f);
  } else if (type == "matmul") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("use_quantizer", use_quantizer);
    op->SetAttr("Scale_x", 1.0f);
    op->SetAttr("Scale_y", 1.0f);
    op->SetAttr("Scale_out", 1.0f);
  }
}

void InitTensorHolder(Scope* scope, const paddle::platform::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<LoDTensor>();
  tensor->mutable_data(place, proto::VarType::FP32, 1);
}

void PreparePass(std::unique_ptr<ir::Graph>* graph, const ProgramDesc& prog,
                 const std::initializer_list<std::string> variable_names,
                 int* original_nodes_num, int* current_nodes_num,
                 std::string var_without_scale = "") {
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  exe.CreateVariables(prog, 0, true, &scope);
  auto* scales = new VarQuantScale();
  for (auto& v : variable_names) {
    if (v.compare(var_without_scale) == 0) continue;
    InitTensorHolder(&scope, place, v.c_str());
    LoDTensor tensor;
    tensor.Resize({1});
    auto* ptr = tensor.mutable_data<double>(place);
    ptr[0] = 2.0;

    (*scales)[v] = std::make_pair(false, std::move(tensor));
  }

  (*graph)->SetNotOwned(kParamScopeAttr, &scope);
  std::unique_ptr<Pass> pass =
      PassRegistry::Instance().Get("cpu_quantize_pass");
  pass->Set("quant_var_scales", scales);

  *original_nodes_num = (*graph)->Nodes().size();
  (*graph).reset(pass->Apply((*graph).release()));
  *current_nodes_num = (*graph)->Nodes().size();
}

namespace {
static const std::initializer_list<std::string> variable_names{
    "a",  "w1", "c", "d", "w2", "e",  "f",  "g", "h",
    "w3", "b1", "i", "j", "w4", "b2", "w5", "b3"};
// (a,w1)->Conv1->c and c->Pool1->d
//
// (d,w2)->Conv2->e and e->Pool2->f
//
// d->Dropout1->g and (g, w5, b3)->Fc1->h and (h,w3,b1,i)->Conv3->j
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
  SetOp(&prog, "fc", "Fc1", {"g", "w5", "b3"}, {"h"}, use_mkldnn,
        use_quantizer);
  SetOp(&prog, "conv2d", "Conv3", {"h", "w3", "b1", "i"}, {"j"}, use_mkldnn,
        use_quantizer);

  SetOp(&prog, "conv2d", "Conv4", {"c", "w4", "b2"}, {"i"}, use_mkldnn,
        use_quantizer);

  return prog;
}

void MainTest(const ProgramDesc& prog, int conv_count, int pool_count,
              int quant_count, int dequant_count, int added_nodes_count,
              float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int conv2d_nodes_count = 0;
  int pool2d_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "conv2d") {
        conv2d_nodes_count++;
        auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_in")), scale)
            << "Scale_in for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_out")), scale)
            << "Scale_out for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(std::vector<float>,
                                  op->GetAttr("Scale_weights"))[0],
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
  // d->Dropout1->g and (g->QUANT8->IN8,w5,b3)->Fc1->OUT7->DEQUANT7->h and
  // (h->QUANT5->IN5,w3,b1,i->QUANT6->IN6)->Conv3->OUT5->DEQUANT5->j
  //
  // (d->QUANT7->IN7,w4, b2)->Conv4->DEQUANT6->OUT6->i
  // Insert nodes: 8 Quant + 8 IN + 7 OUT + 7 DEQUANT
  int added_nodes = 8 + 8 + 7 + 7;
  MainTest(BuildProgramDesc(use_mkldnn, use_quantizer), 4, 2, 8, 7, added_nodes,
           2.0f * 127);
}

TEST(CpuQuantizePass, do_not_quantize) {
  bool use_mkldnn = true;
  bool use_quantizer = false;
  int added_nodes = 0;
  MainTest(BuildProgramDesc(use_mkldnn, use_quantizer), 4, 2, 0, 0, added_nodes,
           1.0f);
}

static const std::initializer_list<std::string> variable_names_concat = {
    "a1", "b1", "a2", "b2", "c", "d"};

// a1->Pool1->b1
// a2->Pool2->b2
// (b1,b2)->Concat->c
// c->Pool3->d
ProgramDesc BuildProgramDescConcat() {
  ProgramDesc prog;

  SetOp(&prog, "pool2d", "Pool1", {"a1"}, {"b1"}, true, false);
  SetOp(&prog, "pool2d", "Pool2", {"a2"}, {"b2"}, true, false);
  SetOp(&prog, "concat", "Concat", {"b1", "b2"}, {"c"}, true, true);
  SetOp(&prog, "pool2d", "Pool3", {"c"}, {"d"}, true, false);

  return prog;
}

void MainTestConcat(const ProgramDesc& prog, int pool_count, int concat_count,
                    int quant_count, int dequant_count, int added_nodes_count) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names_concat, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int concat_nodes_count = 0;
  int pool2d_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "concat") {
        concat_nodes_count++;
      } else if (op->Type() == "pool2d") {
        pool2d_nodes_count++;
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }
  EXPECT_EQ(concat_nodes_count, concat_count);
  EXPECT_EQ(pool2d_nodes_count, pool_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, concat) {
  // a1->Pool1->b1
  // a2->Pool2->b2
  // (b1->QUANT1->IN1, b2->QUANT2->IN2)->Concat->c
  // c->OUT1->DEQUANT1->Pool3->d
  int pool_count = 3;
  int concat_count = 1;
  int quant_count = 2;
  int dequant_count = 1;
  int added_nodes_count = 6;
  MainTestConcat(BuildProgramDescConcat(), pool_count, concat_count,
                 quant_count, dequant_count, added_nodes_count);
}

static const std::initializer_list<std::string> variable_names_transpose = {
    "a", "w1", "b", "c", "w2", "d", "e", "f"};

// a->Conv1->b
// b->Transpose1->c
// c->Conv2->d
// d->Transpose2->e
// e->Dropout->f
ProgramDesc BuildProgramDescTranspose() {
  ProgramDesc prog;
  for (auto& v : variable_names_transpose) {
    auto* var = prog.MutableBlock(0)->Var(v);
    if (v.find("w") == 0) {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "conv2d", "Conv1", {"a", "w1"}, {"b"}, true, true);
  SetOp(&prog, "transpose2", "Transpose1", {"b"}, {"c"}, true, true);
  SetOp(&prog, "conv2d", "Conv1", {"c", "w2"}, {"d"}, true, true);
  SetOp(&prog, "transpose2", "Transpose2", {"d"}, {"e"}, true, true);
  SetOp(&prog, "dropout", "Dropout", {"e"}, {"f"}, true, false);

  return prog;
}

void MainTestTranspose(const ProgramDesc& prog, int conv_count,
                       int transpose_count, int quant_count, int dequant_count,
                       int added_nodes_count, float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names_transpose, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int transpose_nodes_count = 0;
  int conv_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "transpose2") {
        transpose_nodes_count++;
      } else if (op->Type() == "conv2d") {
        conv_nodes_count++;
        auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_in")), scale)
            << "Scale_in for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_out")), scale)
            << "Scale_out for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(std::vector<float>,
                                  op->GetAttr("Scale_weights"))[0],
                  scale)
            << "Scale_weights for node '" + op_name + "'.";
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }
  EXPECT_EQ(transpose_nodes_count, transpose_count);
  EXPECT_EQ(conv_nodes_count, conv_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, transpose) {
  // a1->Quant->a2->Conv1->b1->Dequant->b2
  // b2->Quant->b3->Transpose->c1->Dequant->c2
  // c2->Quant->c3->Conv2->d1->Dequant->d2
  // d2->Quant->d3->Transpose->e1->Dequant->e2
  // e2->Dropout->f
  int conv_count = 2;
  int transpose_count = 2;
  int quant_count = 4;
  int dequant_count = 4;
  // 4 Quant + 4 IN + 4 DeQuant + 4 OUT
  int added_nodes_count = 4 + 4 + 4 + 4;
  MainTestTranspose(BuildProgramDescTranspose(), conv_count, transpose_count,
                    quant_count, dequant_count, added_nodes_count, 2.0f * 127);
}

static const std::initializer_list<std::string> variable_names_reshape = {
    "a", "w1", "b", "c", "d", "e", "f"};

// a->Dequantize->b
// b->Reshape->c
// c->Dropout->d
ProgramDesc BuildProgramDescReshape() {
  ProgramDesc prog;
  for (auto& v : variable_names_transpose) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, "reshape2", "Reshape2", {"b"}, {"c"}, true, true);
  SetOp(&prog, "dropout", "Dropout", {"c"}, {"d"}, true, false);

  return prog;
}

// a->Transpose->b
// b->Reshape->c
// c->Dropout->d
ProgramDesc BuildProgramDescReshapeBetweenNonQuantizedOp() {
  ProgramDesc prog;
  for (auto& v : variable_names_transpose) {
    prog.MutableBlock(0)->Var(v);
  }

  SetOp(&prog, "transpose2", "Transpose2", {"a"}, {"b"}, true, false);
  SetOp(&prog, "reshape2", "Reshape2", {"b"}, {"c"}, true, true);
  SetOp(&prog, "dropout", "Dropout", {"c"}, {"d"}, true, false);

  return prog;
}

void MainTestReshape(const ProgramDesc& prog, int transpose_count,
                     int reshape_count, int quant_count, int dequant_count,
                     int added_nodes_count, float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names_reshape, &original_nodes_num,
              &current_nodes_num);

  float quant_scale = 1.0f;
  float dequant_scale = 1.0f;
  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int transpose_nodes_count = 0;
  int reshape_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "transpose2") {
        transpose_nodes_count++;
      } else if (op->Type() == "reshape2") {
        reshape_nodes_count++;
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
        quant_scale = BOOST_GET_CONST(float, op->GetAttr("Scale"));
        EXPECT_EQ(quant_scale, scale) << "Scale for node '" + op->Type() + "'.";
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
        auto op_name = op->GetAttrIfExists<std::string>("name");
        VLOG(3) << op_name << "\n";
        if (op_name != "Dequantize1") {
          dequant_scale = BOOST_GET_CONST(float, op->GetAttr("Scale"));
          EXPECT_EQ(dequant_scale, scale)
              << "Scale for node '" + op->Type() + "'.";
        }
      }
    }
  }
  EXPECT_EQ(transpose_nodes_count, transpose_count);
  EXPECT_EQ(reshape_nodes_count, reshape_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, reshape) {
  // a->Dequantize->b
  // b2->Quant->b3->Reshape2->c1->Dequant->c2
  // c2->Dropout->d
  int reshape_count = 1;
  int transpose_count = 0;
  int quant_count = 1;
  int dequant_count = 2;
  // 1 Quant + 1 IN + 1 DeQuant + 1 OUT
  int added_nodes_count = 4;
  MainTestReshape(BuildProgramDescReshape(), transpose_count, reshape_count,
                  quant_count, dequant_count, added_nodes_count, 2.0f * 127);
}

TEST(CpuQuantizePass, reshapeBetweenNonQuantizedOp) {
  // a->Transpos2->b
  // b->Reshape2->c
  // c->Dropout->d
  int reshape_count = 1;
  int transpose_count = 1;
  int quant_count = 0;
  int dequant_count = 0;
  // 0 Quant + 0 IN + 0 DeQuant + 0 OUT
  int added_nodes_count = 0;
  MainTestReshape(BuildProgramDescReshapeBetweenNonQuantizedOp(),
                  transpose_count, reshape_count, quant_count, dequant_count,
                  added_nodes_count, 2.0f * 127);
}

static const std::initializer_list<std::string> variable_names_matmul = {
    "a", "b", "c", "d", "e", "f"};

ProgramDesc BuildProgramDescMatmul() {
  ProgramDesc prog;
  for (auto& v : variable_names_transpose) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dequantize", "Dequantize1", {"a"}, {"b"}, true);
  SetOp(&prog, "dequantize", "Dequantize2", {"c"}, {"d"}, true);
  SetOp(&prog, "matmul", "Matmul", {"b", "d"}, {"e"}, true, true);
  SetOp(&prog, "dropout", "Dropout", {"e"}, {"f"}, true, false);

  return prog;
}

ProgramDesc BuildProgramDescMatmulNotQuantized() {
  ProgramDesc prog;
  for (auto& v : variable_names_transpose) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dropout", "Dropout", {"a"}, {"b"}, false);
  SetOp(&prog, "dequantize", "Dequantize", {"c"}, {"d"}, true);
  SetOp(&prog, "matmul", "Matmul", {"b", "d"}, {"e"}, true, true);
  SetOp(&prog, "dropout", "Dropout", {"e"}, {"f"}, true, false);

  return prog;
}

void MainTestMatmul(const ProgramDesc& prog, int matmul_count, int quant_count,
                    int dequant_count, int added_nodes_count, float scale) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names_matmul, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int matmul_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "matmul") {
        matmul_nodes_count++;
        auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_x")), scale)
            << "Scale_x for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_y")), scale)
            << "Scale_y for node '" + op_name + "'.";
        EXPECT_EQ(BOOST_GET_CONST(float, op->GetAttr("Scale_out")), scale)
            << "Scale_out for node '" + op_name + "'.";
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }
  EXPECT_EQ(matmul_nodes_count, matmul_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, matmul) {
  int matmul_count = 1;
  int quant_count = 2;
  int dequant_count = 3;
  // 2 Quant + 2 IN + 1 DeQuant + 1 OUT
  int added_nodes_count = 6;
  MainTestMatmul(BuildProgramDescMatmul(), matmul_count, quant_count,
                 dequant_count, added_nodes_count, 2.0f * 127);
}

TEST(CpuQuantizePass, matmul_not_quantized) {
  int matmul_count = 1;
  int quant_count = 0;
  int dequant_count = 1;
  // nothing change
  int added_nodes_count = 0;
  MainTestMatmul(BuildProgramDescMatmulNotQuantized(), matmul_count,
                 quant_count, dequant_count, added_nodes_count, 1.0f);
}
}  // namespace

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_quantize_pass);
