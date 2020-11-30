// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn,
           const std::string& mkldnn_data_type = "float32",
           const bool force_fp32_output = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);

  if (type == "conv2d") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Output", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
    op->SetAttr("force_fp32_output", force_fp32_output);
  } else if (type == "pool2d" || type == "transpose2" || type == "reshape2" ||
             type == "dropout") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
    if (type != "dropout") op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  } else if (type == "fc") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
    op->SetAttr("force_fp32_output", force_fp32_output);
  } else if (type == "concat" || type == "sum") {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  } else if (type == "matmul" || type == "elementwise_add" ||
             type == "elementwise_mul") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
    if (type == "matmul") op->SetAttr("force_fp32_output", force_fp32_output);
  } else if (type == "layer_norm") {
    op->SetInput("X", {inputs[0]});
    op->SetOutput("Y", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  }
}

static const std::initializer_list<std::string> variable_names{
    "z", "a", "b", "c", "d", "e", "f", "g", "h", "i"};

void PreparePass(std::unique_ptr<ir::Graph>* graph, const ProgramDesc& prog,
                 const std::initializer_list<std::string> variable_names,
                 int* original_nodes_num, int* current_nodes_num) {
  auto pass = PassRegistry::Instance().Get("cpu_bfloat16_pass");

  *original_nodes_num = (*graph)->Nodes().size();
  (*graph).reset(pass->Apply((*graph).release()));
  *current_nodes_num = (*graph)->Nodes().size();
}

void MainTest(const ProgramDesc& prog, int quant_count, int dequant_count,
              int force_fp32_count, int added_nodes_count) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int force_fp32_nodes_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      } else if (op->Type() == "conv2d" || op->Type() == "matmul" ||
                 op->Type() == "fc") {
        if (op->GetAttrIfExists<bool>("force_fp32_output"))
          force_fp32_nodes_count++;
      }
    }
  }
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(force_fp32_nodes_count, force_fp32_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

ProgramDesc BuildProgramDescConv(bool use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dropout", "Dropout", {"a"}, {"b"}, use_mkldnn, "float32");
  SetOp(&prog, "conv2d", "Conv1", {"b"}, {"c"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "pool2d", "Pool", {"c"}, {"d"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "conv2d", "Conv2", {"d"}, {"e"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "transpose2", "Transpose", {"e"}, {"f"}, use_mkldnn, "float32");

  return prog;
}

TEST(CpuBfloat16Pass, convolution) {
  bool use_mkldnn = true;
  // 0 added + 1 force_fp32_output
  int added_nodes = 0;
  MainTest(BuildProgramDescConv(use_mkldnn), 0, 0, 1, added_nodes);
}

ProgramDesc BuildProgramDescDoubleInput(bool use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dropout", "Dropout", {"a"}, {"b"}, use_mkldnn, "float32");
  SetOp(&prog, "matmul", "Matmul", {"b", "b"}, {"c"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "transpose2", "Transpose", {"d"}, {"e"}, use_mkldnn, "float32");
  SetOp(&prog, "elementwise_add", "ElemetwiseAdd", {"c", "e"}, {"f"},
        use_mkldnn, "bfloat16");
  SetOp(&prog, "reshape2", "Reshape", {"f"}, {"g"}, use_mkldnn, "bfloat16");

  return prog;
}

TEST(CpuBfloat16Pass, double_input_ops) {
  bool use_mkldnn = true;
  // 2 quant + 2 quant out
  int added_nodes = 4;
  MainTest(BuildProgramDescDoubleInput(use_mkldnn), 2, 0, 0, added_nodes);
}

ProgramDesc BuildProgramDescDuplicatedInput(bool use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dropout", "Dropout1", {"a"}, {"b"}, use_mkldnn, "float32");
  SetOp(&prog, "dropout", "Dropout2", {"c"}, {"d"}, use_mkldnn, "float32");
  SetOp(&prog, "concat", "Concat", {"b", "d"}, {"e"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "transpose2", "Transpose", {"f"}, {"g"}, use_mkldnn, "float32");
  SetOp(&prog, "sum", "Sum", {"e", "g"}, {"h"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "reshape2", "Reshape", {"h"}, {"i"}, use_mkldnn, "bfloat16");

  return prog;
}

TEST(CpuBfloat16Pass, duplicated_input_ops) {
  bool use_mkldnn = true;
  // 3 quant + 3 quant out
  int added_nodes = 6;
  MainTest(BuildProgramDescDuplicatedInput(use_mkldnn), 3, 0, 0, added_nodes);
}

ProgramDesc BuildProgramDescDoubleOutputs(bool use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "layer_norm", "LayerNorm1", {"a"}, {"b"}, use_mkldnn,
        "bfloat16");
  SetOp(&prog, "dropout", "Dropout1", {"b"}, {"c"}, use_mkldnn, "float32");
  SetOp(&prog, "transpose2", "Transpose", {"b"}, {"d"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "layer_norm", "LayerNorm2", {"d"}, {"e"}, use_mkldnn,
        "bfloat16");
  SetOp(&prog, "reshape2", "Reshape", {"e"}, {"f"}, use_mkldnn, "float32");
  SetOp(&prog, "dropout", "Dropout2", {"e"}, {"g"}, use_mkldnn, "float32");

  return prog;
}

TEST(CpuBfloat16Pass, double_outputs_ops) {
  bool use_mkldnn = true;
  // 3 dequant + 3 dequant out
  int added_nodes = 6;
  MainTest(BuildProgramDescDoubleOutputs(use_mkldnn), 0, 3, 0, added_nodes);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_bfloat16_pass);
