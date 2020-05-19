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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_placement_pass.h"

#include <gtest/gtest.h>
#include <boost/logic/tribool.hpp>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs,
           boost::tribool use_quantizer) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);

  if (!boost::indeterminate(use_quantizer))
    op->SetAttr("use_quantizer", use_quantizer);

  if (type == "conv2d") {
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {inputs[2]});
  } else if (type == "relu") {
    op->SetInput("X", inputs);
  } else if (type == "concat") {
    op->SetAttr("axis", 1);
    op->SetInput("X", {inputs[0], inputs[1]});
  } else if (type == "pool2d") {
    op->SetInput("X", {inputs[0]});
  } else {
    FAIL() << "Unexpected operator type.";
  }
  op->SetOutput("Out", {outputs[0]});
}

// operator                      use_quantizer
// ---------------------------------------
// (a,b)->concat->c              none
// (c,weights,bias)->conv->f     false
// f->relu->g                    none
// g->pool->h                    false
// (h,weights2,bias2)->conv->k   false
// k->pool->l                    false
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;

  for (auto& v :
       std::vector<std::string>({"a", "b", "c", "weights", "bias", "f", "g",
                                 "h", "weights2", "bias2", "k", "l"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  SetOp(&prog, "concat", "concat1", {"a", "b"}, {"c"}, boost::indeterminate);
  SetOp(&prog, "conv2d", "conv1", {"c", "weights", "bias"}, {"f"}, false);
  SetOp(&prog, "relu", "relu1", {"f"}, {"g"}, boost::indeterminate);
  SetOp(&prog, "pool2d", "pool1", {"g"}, {"h"}, false);
  SetOp(&prog, "conv2d", "conv2", {"h", "weights2", "bias2"}, {"k"}, false);
  SetOp(&prog, "pool2d", "pool2", {"k"}, {"l"}, false);

  return prog;
}

void MainTest(std::initializer_list<std::string> quantize_enabled_op_types,
              std::initializer_list<int> quantize_excluded_op_ids,
              unsigned expected_use_quantizer_true_count) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("cpu_quantize_placement_pass");
  pass->Set("quantize_enabled_op_types",
            new std::unordered_set<std::string>(quantize_enabled_op_types));
  pass->Set("quantize_excluded_op_ids",
            new std::unordered_set<int>(quantize_excluded_op_ids));

  graph.reset(pass->Apply(graph.release()));

  unsigned use_quantizer_true_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->HasAttr("use_quantizer") &&
          BOOST_GET_CONST(bool, op->GetAttr("use_quantizer"))) {
        ++use_quantizer_true_count;
      }
    }
  }

  EXPECT_EQ(use_quantizer_true_count, expected_use_quantizer_true_count);
}

void DefaultAttrTest(unsigned expected_use_quantizer_true_count) {
  auto prog = BuildProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto pass = PassRegistry::Instance().Get("cpu_quantize_placement_pass");
  graph.reset(pass->Apply(graph.release()));

  unsigned use_quantizer_true_count = 0;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->HasAttr("use_quantizer") &&
          BOOST_GET_CONST(bool, op->GetAttr("use_quantizer"))) {
        ++use_quantizer_true_count;
      }
    }
  }
  EXPECT_EQ(use_quantizer_true_count, expected_use_quantizer_true_count);
}

TEST(QuantizerPlacementPass, enabled_pool) { MainTest({"pool2d"}, {}, 2); }

TEST(QuantizerPlacementPass, enabled_conv_excluded_one) {
  MainTest({"conv2d"}, {4}, 1);
}

TEST(QuantizerPlacementPass, excluded_none) {
  // 2 conv + 2 pool
  MainTest({}, {}, 4);
}

TEST(QuantizerPlacementPass, default_attr_value) {
  // 2 conv + 2 pool
  DefaultAttrTest(4);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(cpu_quantize_placement_pass);
