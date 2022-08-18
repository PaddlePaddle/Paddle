// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/matmul_activation_mkldnn_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void MatmulActivationMkldnnFusePass::ApplyImpl(Graph* graph) const {
  auto act_types = paddle::platform::GetSupportedActivations();
  auto matmul_types = {"matmul", "matmul_v2"};

  for (const auto& matmul_type : matmul_types)
    for (auto& act_type : act_types) {
      FuseMatmulAct(graph, matmul_type, act_type);
    }
}

void MatmulActivationMkldnnFusePass::FuseMatmulAct(
    Graph* graph, const std::string& matmul_type, std::string& act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(matmul_type + "_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation matmul_act_pattern(
      gpd.mutable_pattern(), "matmul_activation_mkldnn_fuse");
  matmul_act_pattern(matmul_type, act_type);

  int found_matmul_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle " + matmul_type + "+" + act_type + " fuse";

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "matmul_activation_mkldnn_fuse_pass op compat failed.";
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(matmul, preceding_op, matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, preceding_op_out, matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, matmul_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        activation_out, activation_out, matmul_act_pattern);

    OpDesc* matmul_op = matmul->Op();
    OpDesc* act_op = activation->Op();

    auto attr_map = paddle::platform::GetAttributeMap(act_type);
    for (const auto& attrs : attr_map) {
      if (act_op->HasAttr(attrs.first)) {
        matmul_op->SetAttr(attrs.second, act_op->GetAttr(attrs.first));
      }
    }

    if (act_type == "gelu" && activation->Op()->HasAttr("approximate")) {
      act_type =
          PADDLE_GET_CONST(bool, activation->Op()->GetAttr("approximate"))
              ? "gelu_tanh"
              : "gelu_erf";
    }
    matmul_op->SetAttr("fuse_activation", act_type);
    matmul_op->SetOutput("Out", {activation_out->Name()});

    IR_NODE_LINK_TO(matmul, activation_out);
    GraphSafeRemoveNodes(graph, {activation, matmul_out});
    found_matmul_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_matmul_activation_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    PrettyLogDetail("---    fused %d %s with %s activation",
                    found_matmul_activation_count,
                    matmul_type,
                    act_type);
  }
}

MatmulActivationMkldnnFusePass::MatmulActivationMkldnnFusePass() {
  AddOpCompat(OpCompat("matmul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddInput(
          "ResidualData")  // Extra tensor used in matmul+elementwise_add fuse
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsType<float>()
      .End()
      .AddAttr("transpose_X")
      .IsType<bool>()
      .End()
      .AddAttr("transpose_Y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("matmul_v2"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddInput(
          "ResidualData")  // Extra tensor used in matmul+elementwise_add fuse
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("trans_x")
      .IsType<bool>()
      .End()
      .AddAttr("trans_y")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("abs"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("clip"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("min")
      .End()
      .AddAttr("max")
      .End();

  AddOpCompat(OpCompat("gelu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("approximate")
      .IsType<bool>()
      .End();

  AddOpCompat(OpCompat("hard_sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("slope")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("hard_swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("threshold")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("scale")
      .IsOptional()
      .IsType<float>()
      .End()
      .AddAttr("offset")
      .IsOptional()
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("leaky_relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("alpha")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("mish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("relu6"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("threshold")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("sigmoid"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("sqrt"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  AddOpCompat(OpCompat("swish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("beta")
      .IsType<float>()
      .End();

  AddOpCompat(OpCompat("tanh"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(matmul_activation_mkldnn_fuse_pass,
              paddle::framework::ir::MatmulActivationMkldnnFusePass);

REGISTER_PASS_CAPABILITY(matmul_activation_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("matmul", 1)
            .EQ("matmul_v2", 0)
            .EQ("abs", 0)
            .LE("clip", 1)
            .EQ("gelu", 0)
            .EQ("hard_sigmoid", 0)
            .LE("hard_swish", 0)
            .LE("leaky_relu", 1)
            .LE("mish", 1)
            .EQ("relu", 0)
            .EQ("relu6", 0)
            .EQ("sigmoid", 0)
            .EQ("sqrt", 0)
            .EQ("swish", 0)
            .EQ("tanh", 0));
