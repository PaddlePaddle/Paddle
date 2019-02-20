// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_bias_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/platform/place.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/op_proto_maker.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv2d") {
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("name", name);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    op->SetInput("Bias", {});
  } else if (type == "elementwise_add") {
    op->SetAttr("use_mkldnn", true);
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
  }
  op->SetOutput("Out", outputs);
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// (c, weights)->conv->f
// (f)->elementwise_add->g
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v :
       std::vector<std::string>({"c", "weights", "f", "bias", "g"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
    if (v == "weights" || v == "bias") {
      var->SetPersistable(true);
    }
  }

  // conv+bias, both with MKL-DNN
  SetOp(&prog, "conv2d", "conv",
        std::vector<std::string>({"c", "weights"}),
        std::vector<std::string>({"f"}));
  SetOp(&prog, "elementwise_add", "eltwise",
        std::vector<std::string>({"f", "bias"}),
        std::vector<std::string>({"g"}));

  return prog;
}

TEST(ConvBiasFusePass, basic) {
  auto prog = BuildProgramDesc();
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  NaiveExecutor exe{paddle::platform::CPUPlace()};
  Scope scope;
  // Init scope, as it is used in pass
  exe.CreateVariables(prog, 0, true, &scope);
  graph->Set(kParamScopeAttr, new framework::Scope*(
                    const_cast<framework::Scope*>(&scope)));

  auto pass = PassRegistry::Instance().Get("conv_bias_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph = pass->Apply(std::move(graph));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: Conv, Bias, conv_out
  // Add 1 Node: ConvBias
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_bias op in newly generated graph
  int conv_bias_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(boost::get<bool>(op->GetAttr("use_mkldnn")));
      // check if "conv" convolution is fused
      auto op_name = boost::get<std::string>(op->GetAttr("name"));
      if (op_name == "conv") {
        auto input_names = op->InputNames();
        ASSERT_TRUE(std::find(input_names.begin(), input_names.end(), "Bias")
                    != input_names.end());
        auto bias = boost::get<std::vector<std::string>>(op->Input("Bias"));
        if (bias.size()) {
          ++conv_bias_count;
        }
      }
    }
  }
  EXPECT_EQ(conv_bias_count, 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_bias_mkldnn_fuse_pass);
