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
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  } else if (type == "fc") {
    op->SetInput("Input", {inputs[0]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  } else if (type == "concat") {
    op->SetInput("X", inputs);
    op->SetOutput("Out", outputs);
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  } else if (type == "matmul" || type == "elementwise_add") {
    op->SetInput("X", {inputs[0]});
    if (inputs.size() > 1) op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", {outputs[0]});
    op->SetAttr("mkldnn_data_type", mkldnn_data_type);
  }
}

void PreparePass(std::unique_ptr<ir::Graph>* graph, const ProgramDesc& prog,
                 const std::initializer_list<std::string> variable_names,
                 int* original_nodes_num, int* current_nodes_num) {
  auto pass = PassRegistry::Instance().Get("cpu_bfloat16_pass");

  graph->reset(pass->Apply(graph->release()));

  *original_nodes_num = (*graph)->Nodes().size();
  (*graph).reset(pass->Apply((*graph).release()));
  *current_nodes_num = (*graph)->Nodes().size();
}

static const std::initializer_list<std::string> variable_names{
    "z", "a", "b", "c", "d", "e", "f", "g", "h", "i"};

ProgramDesc BuildProgramDesc(bool use_mkldnn) {
  ProgramDesc prog;
  for (auto& v : variable_names) {
    prog.MutableBlock(0)->Var(v);
  }
  SetOp(&prog, "dropout", "Dropout1", {"z"}, {"a"}, use_mkldnn, "float32");
  SetOp(&prog, "conv2d", "Conv1", {"a"}, {"b"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "pool2d", "Pool1", {"b"}, {"c"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "conv2d", "Conv1", {"c"}, {"d"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "dropout", "Dropout2", {"d"}, {"e"}, use_mkldnn, "float32");
  SetOp(&prog, "transpose2", "Transpose1", {"e"}, {"f"}, use_mkldnn,
        "bfloat16");
  SetOp(&prog, "reshape2", "Reshape1", {"f"}, {"g"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "concat", "Concat1", {"g"}, {"h"}, use_mkldnn, "bfloat16");
  SetOp(&prog, "dropout", "Dropout3", {"h"}, {"i"}, use_mkldnn, "float32");

  return prog;
}

void MainTest(const ProgramDesc& prog, int conv_count, int pool_count,
              int transpose_count, int quant_count, int dequant_count,
              int added_nodes_count) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  int original_nodes_num, current_nodes_num;
  PreparePass(&graph, prog, variable_names, &original_nodes_num,
              &current_nodes_num);

  int quantize_nodes_count = 0;
  int dequantize_nodes_count = 0;
  int conv2d_nodes_count = 0;
  int pool2d_nodes_count = 0;
  int transpose2_nodes_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "conv2d") {
        conv2d_nodes_count++;
      } else if (op->Type() == "pool2d") {
        pool2d_nodes_count++;
      } else if (op->Type() == "transpose2") {
        transpose2_nodes_count++;
      } else if (op->Type() == "quantize") {
        quantize_nodes_count++;
      } else if (op->Type() == "dequantize") {
        dequantize_nodes_count++;
      }
    }
  }
  EXPECT_EQ(conv2d_nodes_count, conv_count);
  EXPECT_EQ(pool2d_nodes_count, pool_count);
  EXPECT_EQ(transpose2_nodes_count, transpose_count);
  EXPECT_EQ(quantize_nodes_count, quant_count);
  EXPECT_EQ(dequantize_nodes_count, dequant_count);
  EXPECT_EQ(original_nodes_num + added_nodes_count, current_nodes_num);
}

TEST(CpuQuantizePass, quantize) {
  bool use_mkldnn = true;
  // 1 quantize + 1 dequantize
  int added_nodes = 2;
  MainTest(BuildProgramDesc(use_mkldnn), 2, 1, 1, 1, 2, added_nodes);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_bfloat16_pass);
