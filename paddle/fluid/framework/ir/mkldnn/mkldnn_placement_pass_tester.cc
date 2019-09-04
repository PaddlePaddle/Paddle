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

#include "paddle/fluid/framework/ir/mkldnn/mkldnn_placement_pass.h"

#include <gtest/gtest.h>
#include <boost/logic/tribool.hpp>

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, boost::tribool use_mkldnn) {
  auto* op = prog->MutableBlock(0)->AppendOp();

  op->SetType(type);

  if (!boost::indeterminate(use_mkldnn)) op->SetAttr("use_mkldnn", use_mkldnn);

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

// operator                      use_mkldnn
// ---------------------------------------
// (a,b)->concat->c              none
// (c,weights,bias)->conv->f     none
// f->relu->g                    false
// g->pool->h                    false
// (h,weights2,bias2)->conv->k   true
// k->relu->l                    true
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

  SetOp(&prog, "concat", "concat1", std::vector<std::string>({"a", "b"}),
        std::vector<std::string>({"c"}), boost::indeterminate);
  SetOp(&prog, "conv2d", "conv1",
        std::vector<std::string>({"c", "weights", "bias"}),
        std::vector<std::string>({"f"}), boost::indeterminate);
  SetOp(&prog, "relu", "relu1", std::vector<std::string>({"f"}),
        std::vector<std::string>({"g"}), false);
  SetOp(&prog, "pool2d", "pool1", std::vector<std::string>({"g"}),
        std::vector<std::string>({"h"}), false);
  SetOp(&prog, "conv2d", "conv2",
        std::vector<std::string>({"h", "weights2", "bias2"}),
        std::vector<std::string>({"k"}), true);
  SetOp(&prog, "relu", "relu2", std::vector<std::string>({"k"}),
        std::vector<std::string>({"l"}), true);

  return prog;
}

void MainTest(std::initializer_list<std::string> mkldnn_enabled_op_types,
              unsigned expected_use_mkldnn_true_count) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("mkldnn_placement_pass");
  pass->Set("mkldnn_enabled_op_types",
            new std::unordered_set<std::string>(mkldnn_enabled_op_types));

  graph.reset(pass->Apply(graph.release()));

  unsigned use_mkldnn_true_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->HasAttr("use_mkldnn") &&
          boost::get<bool>(op->GetAttr("use_mkldnn"))) {
        ++use_mkldnn_true_count;
      }
    }
  }

  EXPECT_EQ(use_mkldnn_true_count, expected_use_mkldnn_true_count);
}

TEST(MKLDNNPlacementPass, enable_conv_relu) {
  // 1 conv (1 conv is always true) + 2 relu (1 relu is always true) + 0 pool
  MainTest({"conv2d", "relu"}, 3);
}

TEST(MKLDNNPlacementPass, enable_relu_pool) {
  // 1 conv (1 conv is always true) + 2 relu (1 relu is always true) + 1 pool
  MainTest({"relu", "pool2d"}, 4);
}

TEST(MKLDNNPlacementPass, enable_all) {
  // 1 conv (1 conv is always true) + 2 relu (1 relu is always true) + 1 pool
  MainTest({}, 4);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(mkldnn_placement_pass);
