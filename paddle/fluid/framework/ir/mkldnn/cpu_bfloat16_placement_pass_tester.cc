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

#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_placement_pass.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           const std::string& mkldnn_data_type = "float32",
           const bool use_mkldnn = true) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);
  if (type != "reshape2") op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("mkldnn_data_type", mkldnn_data_type);

  if (type == "conv2d") {
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
  } else if (type == "gelu") {
    op->SetInput("X", inputs);
  } else if (type == "concat") {
    op->SetAttr("axis", 1);
    op->SetInput("X", {inputs[0], inputs[1]});
  } else if (type == "pool2d") {
    op->SetInput("X", {inputs[0]});
  } else if (type == "transpose2") {
    op->SetInput("X", {inputs[0]});
  } else if (type == "reshape2") {
    op->SetInput("X", {inputs[0]});
  } else if (type == "sum") {
    op->SetInput("X", {inputs[0], inputs[1]});
  } else {
    FAIL() << "Unexpected operator type.";
  }
  op->SetOutput("Out", {outputs[0]});
}

// operator                      mkldnn_data_type
// ---------------------------------------
// (a,b)->concat->c              float32
// c->conv->f                    float32
// f->relu->g                    float32
// g->pool->h                    float32
// h->conv->k                    float32
// k->pool->l                    float32
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;

  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "f", "g", "h", "k", "l", "m",
                                 "n", "o", "p", "r", "s"})) {
    prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::FP32);
  }

  SetOp(&prog, "concat", "concat1", {"a", "b"}, {"c"});
  SetOp(&prog, "conv2d", "conv1", {"c"}, {"f"});
  SetOp(&prog, "gelu", "gelu1", {"f"}, {"g"});
  SetOp(&prog, "pool2d", "pool1", {"g"}, {"h"});
  SetOp(&prog, "conv2d", "conv2", {"h"}, {"k"});
  SetOp(&prog, "pool2d", "pool2", {"k"}, {"l"});
  SetOp(&prog, "concat", "concat2", {"l", "m"}, {"n"});
  SetOp(&prog, "transpose2", "transpose", {"n"}, {"o"});
  SetOp(&prog, "reshape2", "reshape", {"o"}, {"p"});
  SetOp(&prog, "sum", "sum", {"p", "r"}, {"s"});

  return prog;
}

void MainTest(std::initializer_list<std::string> bfloat16_enabled_op_types,
              unsigned expected_bfloat16_data_type_count,
              const ProgramDesc& prog) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("cpu_bfloat16_placement_pass");
  pass->Set("bfloat16_enabled_op_types",
            new std::unordered_set<std::string>(bfloat16_enabled_op_types));

  graph.reset(pass->Apply(graph.release()));

  unsigned bfloat16_data_type_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      if (platform::HasOpBFLOAT16DataType(node->Op())) {
        ++bfloat16_data_type_count;
      }
    }
  }

  EXPECT_EQ(bfloat16_data_type_count, expected_bfloat16_data_type_count);
}

void DefaultAttrTest(unsigned expected_bfloat16_data_type_count,
                     const ProgramDesc& prog) {
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("cpu_bfloat16_placement_pass");
  graph.reset(pass->Apply(graph.release()));

  unsigned bfloat16_data_type_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      if (platform::HasOpBFLOAT16DataType(node->Op())) {
        ++bfloat16_data_type_count;
      }
    }
  }
  EXPECT_EQ(bfloat16_data_type_count, expected_bfloat16_data_type_count);
}

TEST(Bfloat16PlacementPass, enable_all) {
  MainTest({"conv2d", "pool2d", "gelu", "concat", "sum"}, 8,
           BuildProgramDesc());
}

TEST(Bfloat16PlacementPass, enabled_conv_and_pool) {
  // 2 conv2d + 2 pool2 - 1 orphaned conv2d
  MainTest({"conv2d", "pool2d"}, 3, BuildProgramDesc());
}

TEST(Bfloat16PlacementPass, default_attr_value) {
  DefaultAttrTest(10, BuildProgramDesc());
}

ProgramDesc BuildProgramDescWithDataType() {
  ProgramDesc prog;

  for (auto& v : std::vector<std::string>({"a", "b", "c", "d", "e"})) {
    if (v == "a") {
      prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::INT32);
    } else {
      prog.MutableBlock(0)->Var(v)->SetDataType(proto::VarType::FP32);
    }
  }

  SetOp(&prog, "conv2d", "conv1", {"a"}, {"b"});
  SetOp(&prog, "pool2d", "pool1", {"b"}, {"c"});
  SetOp(&prog, "concat", "concat1", {"c", "d"}, {"e"});
  return prog;
}

TEST(Bfloat16PlacementPass, check_data_types) {
  DefaultAttrTest(2, BuildProgramDescWithDataType());
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_bfloat16_placement_pass);
