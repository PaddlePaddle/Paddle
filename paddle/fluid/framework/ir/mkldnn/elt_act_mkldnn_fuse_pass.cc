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
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ElementwiseActivationOneDNNPass::ApplyImpl(Graph *graph) const {
  auto act_types = phi::funcs::GetSupportedActivations();
  std::vector<std::string> elt_types = {
      "elementwise_add", "elementwise_sub", "elementwise_mul"};

  for (const auto &elt_type : elt_types)
    for (const auto &act_type : act_types) {
      FuseElementwiseAct(graph, elt_type, act_type);
    }
}

void ElementwiseActivationOneDNNPass::FuseElementwiseAct(
    Graph *graph,
    const std::string &elt_type,
    const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(elt_type + "_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation elementwise_act_pattern(gpd.mutable_pattern(),
                                                       elt_type + "_act");
  elementwise_act_pattern(elt_type, act_type);

  int found_elementwise_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse " << elt_type << " with activation op.";
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise, preceding_op, elementwise_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        elementwise_out, preceding_op_out, elementwise_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, elementwise_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(
        activation_out, activation_out, elementwise_act_pattern);

    auto *elementwise_op = elementwise->Op();

    if (elementwise_op->HasAttr("use_mkldnn")) {
      const std::string wo_elt_type =
          "The " + elt_type;  // Workaround for PP error message checking.
      PADDLE_ENFORCE_EQ(
          PADDLE_GET_CONST(bool, elementwise_op->GetAttr("use_mkldnn")),
          true,
          platform::errors::PreconditionNotMet(
              wo_elt_type + "+Act fusion may happen only when oneDNN library "
                            "is used."));
    }

    auto *activation_op = activation->Op();
    auto attr_map = phi::funcs::GetAttributeMap(act_type);
    for (const auto &attr : attr_map) {
      if (activation_op->HasAttr(attr.first)) {
        elementwise_op->SetAttr(attr.second,
                                activation_op->GetAttr(attr.first));
      }
    }

    if (act_type == "gelu" && activation_op->HasAttr("approximate") &&
        PADDLE_GET_CONST(bool, activation_op->GetAttr("approximate")))
      elementwise_op->SetAttr("fuse_activation", std::string("gelu_tanh"));
    else
      elementwise_op->SetAttr("fuse_activation", act_type);

    elementwise_op->SetOutput("Out", {activation_out->Name()});

    IR_OP_VAR_LINK(elementwise, activation_out);
    GraphSafeRemoveNodes(g, {activation, elementwise_out});
    found_elementwise_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_elementwise_activation_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      (found_elementwise_activation_count > 0))
    PrettyLogDetail("---    fused %d %s with %s activation",
                    found_elementwise_activation_count,
                    elt_type,
                    act_type);
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
