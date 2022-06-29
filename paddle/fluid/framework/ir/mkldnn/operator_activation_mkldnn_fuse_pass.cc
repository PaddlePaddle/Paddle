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

#include "paddle/fluid/framework/ir/mkldnn/operator_activation_mkldnn_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void OperatorActivationMkldnnFusePass::ApplyImpl(Graph* graph) const {
  std::vector<std::string> op_types = {"conv2d",
                                       "elementwise_add",
                                       "elementwise_sub",
                                       "elementwise_mul",
                                       "matmul",
                                       "softplus"};
  std::vector<std::string> act_types = {"relu",
                                        "mish",
                                        "swish",
                                        "sqrt",
                                        "hard_swish",
                                        "sigmoid",
                                        "abs",
                                        "gelu",
                                        "relu6",
                                        "clip",
                                        "tanh",
                                        "hard_sigmoid",
                                        "leaky_relu"};
  for (const auto& op_type : op_types)
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
      FuseOperatorAct(graph, op_type, act_type, attrs_map);
    }
}

void OperatorActivationMkldnnFusePass::FuseOperatorAct(
    Graph* graph,
    const std::string& op_type,
    std::string& act_type,
    const std::unordered_map<std::string, std::string>& attrs_map) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(op_type + "_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation op_act_pattern(
      gpd.mutable_pattern(), "operator_activation_mkldnn_fuse");
  op_act_pattern(op_type, act_type);

  int found_operator_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(preceding_op, preceding_op, op_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        preceding_op_out, preceding_op_out, op_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, op_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, op_act_pattern);

    OpDesc* activated_op = preceding_op->Op();
    OpDesc* act_op = activation->Op();

    if (activated_op->HasAttr("use_mkldnn")) {
      PADDLE_ENFORCE_EQ(
          BOOST_GET_CONST(bool, activated_op->GetAttr("use_mkldnn")),
          true,
          platform::errors::PreconditionNotMet(
              "Activation fusion may happen only "
              "when oneDNN library is used."));
    }

    for (const auto& attrs : attrs_map) {
      if (act_op->HasAttr(attrs.first)) {
        activated_op->SetAttr(attrs.second, act_op->GetAttr(attrs.first));
      }
    }

    if (act_type == "gelu" && activation->Op()->HasAttr("approximate")) {
      act_type = BOOST_GET_CONST(bool, activation->Op()->GetAttr("approximate"))
                     ? "gelu_tanh"
                     : "gelu_erf";
      activated_op->SetAttr("fuse_alpha", 0.0f);
      activated_op->SetAttr("fuse_beta", 0.0f);
    }

    std::string output_name =
        (op_type == "conv2d" || op_type == "conv2d_transpose") ? "Output"
                                                               : "Out";

    activated_op->SetAttr("fuse_activation", act_type);
    activated_op->SetOutput(output_name, {activation_out->Name()});

    IR_NODE_LINK_TO(preceding_op, activation_out);
    GraphSafeRemoveNodes(graph, {activation, preceding_op_out});
    found_operator_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_operator_activation_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_operator_activation_count > 0) {
    PrettyLogDetail("---    fused %d %s with %s activation",
                    found_operator_activation_count,
                    op_type,
                    act_type);
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(operator_activation_mkldnn_fuse_pass,
              paddle::framework::ir::OperatorActivationMkldnnFusePass);

REGISTER_PASS_CAPABILITY(operator_activation_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("conv2d_transpose", 2)
            .LE("elementwise_add", 1)
            .LE("elementwise_sub", 1)
            .LE("elementwise_mul", 1)
            .LE("matmul", 1)
            .LE("softplus", 1)

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