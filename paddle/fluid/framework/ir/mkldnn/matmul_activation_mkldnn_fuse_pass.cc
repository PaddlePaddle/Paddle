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
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void MatmulActivationMkldnnFusePass::ApplyImpl(Graph* graph) const {
  std::vector<std::string> act_types = {"abs",
                                        "clip",
                                        "gelu",
                                        "hard_sigmoid",
                                        "hard_swish",
                                        "leaky_relu",
                                        "mish",
                                        "relu",
                                        "relu6",
                                        "sigmoid",
                                        "sqrt",
                                        "swish",
                                        "tanh"};

  std::vector<std::string> matmul_types = {"matmul"};

  for (const auto& matmul_type : matmul_types)
    for (auto& act_type : act_types) {
      std::unordered_map<std::string, std::string> attrs_map;

      if (act_type == "swish")
        attrs_map.emplace("beta", "fuse_alpha");
      else if (act_type == "relu6")
        attrs_map.emplace("threshold", "fuse_alpha");
      else if (act_type == "hard_sigmoid") {
        attrs_map.emplace("slope", "fuse_alpha");
        attrs_map.emplace("offset", "fuse_beta");
      } else if (act_type == "clip") {
        attrs_map.emplace("min", "fuse_alpha");
        attrs_map.emplace("max", "fuse_beta");
      } else {
        attrs_map.emplace("alpha", "fuse_alpha");
        attrs_map.emplace("beta", "fuse_beta");
      }
      FuseMatmulAct(graph, matmul_type, act_type, attrs_map);
    }
}

void MatmulActivationMkldnnFusePass::FuseMatmulAct(
    Graph* graph,
    const std::string& matmul_type,
    std::string& act_type,
    const std::unordered_map<std::string, std::string>& attrs_map) const {
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

    for (const auto& attrs : attrs_map) {
      if (act_op->HasAttr(attrs.first)) {
        matmul_op->SetAttr(attrs.second, act_op->GetAttr(attrs.first));
      }
    }

    if (act_type == "gelu" && activation->Op()->HasAttr("approximate")) {
      act_type = BOOST_GET_CONST(bool, activation->Op()->GetAttr("approximate"))
                     ? "gelu_tanh"
                     : "gelu_erf";
      matmul_op->SetAttr("fuse_alpha", 0.0f);
      matmul_op->SetAttr("fuse_beta", 0.0f);
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
    PrettyLogDetail("---    fused %d matmul with %s activation",
                    found_matmul_activation_count,
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

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
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

  AddOpCompat(OpCompat("mish"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
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

  AddOpCompat(OpCompat("tanh"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
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

  AddOpCompat(OpCompat("abs"))
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
