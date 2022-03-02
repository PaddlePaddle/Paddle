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

#include "paddle/fluid/framework/ir/mkldnn/elt_act_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ElementwiseActivationOneDNNPass::ApplyImpl(Graph *graph) const {
  std::vector<std::string> act_types = {
      "relu", "tanh", "leaky_relu", "swish", "hardswish", "sqrt",
      "abs",  "clip", "gelu",       "relu6", "sigmoid"};
  std::vector<std::string> elt_types = {"elementwise_add", "elementwise_sub",
                                        "elementwise_mul"};

  for (const auto &elt_type : elt_types)
    for (const auto &act_type : act_types) {
      std::unordered_map<std::string, std::string> attr_map;

      if (act_type == "swish")
        attr_map.emplace("beta", "activation_alpha");
      else if (act_type == "relu6")
        attr_map.emplace("threshold", "activation_alpha");
      else if (act_type == "clip") {
        attr_map.emplace("min", "activation_alpha");
        attr_map.emplace("max", "activation_beta");
      } else {
        attr_map.emplace("alpha", "activation_alpha");
        attr_map.emplace("beta", "activation_beta");
      }
      FuseElementwiseAct(graph, elt_type, act_type, attr_map);
    }
}

void ElementwiseActivationOneDNNPass::FuseElementwiseAct(
    Graph *graph, const std::string &elt_type, const std::string &act_type,
    const std::unordered_map<std::string, std::string> &attr_map) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("elementwise_act", graph);

  GraphPatternDetector gpd;
  auto *elementwise_input = gpd.mutable_pattern()
                                ->NewNode(elt_type + "_act/elementwise_input")
                                ->AsInput()
                                ->assert_is_op_input(elt_type, "X");
  patterns::ElementwiseActivation elementwise_act_pattern(gpd.mutable_pattern(),
                                                          elt_type + "_act");
  elementwise_act_pattern(elementwise_input, elt_type, act_type);

  int found_elementwise_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse " << elt_type << " with activation op.";
    // Elementwise output
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out,
                              elementwise_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise,
                              elementwise_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, elementwise_act_pattern);

    auto *elementwise_op = elementwise->Op();

    if (elementwise_op->HasAttr("use_mkldnn")) {
      const std::string wo_elt_type =
          "The " + elt_type;  // Workaround for PP error message checking.
      PADDLE_ENFORCE_EQ(
          BOOST_GET_CONST(bool, elementwise_op->GetAttr("use_mkldnn")), true,
          platform::errors::PreconditionNotMet(
              wo_elt_type + "+Act fusion may happen only when oneDNN library "
                            "is used."));
    }

    auto *activation_op = activation->Op();
    for (const auto &attr : attr_map) {
      if (activation_op->HasAttr(attr.first)) {
        elementwise_op->SetAttr(attr.second,
                                activation_op->GetAttr(attr.first));
      }
    }

    if (act_type == "gelu" && activation_op->HasAttr("approximate") &&
        BOOST_GET_CONST(bool, activation_op->GetAttr("approximate")))
      elementwise_op->SetAttr("activation_type", std::string("gelu_tanh"));
    else
      elementwise_op->SetAttr("activation_type", act_type);

    elementwise_op->SetOutput("Out", {activation_out->Name()});

    IR_OP_VAR_LINK(elementwise, activation_out);
    GraphSafeRemoveNodes(g, {activation, elementwise_out});
    found_elementwise_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_elementwise_activation_count);
  PrettyLogDetail("---    fused %d %s with %s activation",
                  found_elementwise_activation_count, elt_type, act_type);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(elt_act_mkldnn_fuse_pass,
              paddle::framework::ir::ElementwiseActivationOneDNNPass);
REGISTER_PASS_CAPABILITY(elt_act_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .LE("elementwise_sub", 1)
            .LE("elementwise_mul", 1)
            .LE("relu", 0)
            .LE("tanh", 0)
            .LE("leaky_relu", 1)
            .LE("swish", 0)
            .LE("hard_swish", 0)
            .LE("sqrt", 0)
            .LE("abs", 0)
            .LE("clip", 1)
            .LE("gelu", 0)
            .LE("relu6", 0)
            .LE("sigmoid", 0));
