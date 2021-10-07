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

#include "paddle/fluid/framework/ir/mkldnn/el_add_act_onednn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void ElementwiseAddActivationOneDNNPass::ApplyImpl(Graph *graph) const {
  std::vector<std::string> act_types = {"relu"};
  std::vector<std::string> elt_types = {"elementwise_add"};

  for (const auto& elt_type : elt_types)
    for (const auto& act_type : act_types)
      FuseElementwiseAddAct(graph, elt_type, act_type);
}

void ElementwiseAddActivationOneDNNPass::FuseElementwiseAddAct(Graph *graph, const std::string &elt_type, const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("elementwise_add_act", graph);

  GraphPatternDetector gpd;
  auto* elementwise_add_input = gpd.mutable_pattern()
                         ->NewNode(elt_type + "_act/elementwise_add_input")
                         ->AsInput()
                         ->assert_is_op_input(elt_type, "X");
  patterns::ElementwiseActivation elementwise_add_act_pattern(
      gpd.mutable_pattern(), elt_type + "_act");
  elementwise_add_act_pattern(elementwise_add_input, elt_type, act_type);

  int found_elementwise_add_activation_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse " << elt_type << " with activation op.";
    // Elementwise Add output
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, elementwise_add_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(activation_out, activation_out, elementwise_add_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, elementwise_add_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(activation, activation, elementwise_add_act_pattern);

    auto *elementwise_add_op = elementwise->Op();

    if (elementwise_add_op->HasAttr("use_mkldnn")) {
      PADDLE_ENFORCE(
          BOOST_GET_CONST(bool, elementwise_add_op->GetAttr("use_mkldnn")),
          platform::errors::PreconditionNotMet(
              "The " + elt_type + "+Act fusion may happen only when oneDNN library "
              "is used."));
    }

    elementwise_add_op->SetAttr("activation_type", act_type);

    elementwise_add_op->SetAttr("use_mkldnn", true);

    elementwise_add_op->SetOutput("Out", {activation_out->Name()});

    IR_OP_VAR_LINK(elementwise, activation_out);
    GraphSafeRemoveNodes(g, {activation, elementwise_out});
    found_elementwise_add_activation_count++;
  };

  gpd(graph, handler);
  AddStatis(found_elementwise_add_activation_count);
  PrettyLogDetail("---    fused %d %s with %s activation", found_elementwise_add_activation_count,
                  elt_type, act_type);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(el_add_act_onednn_fuse_pass,
              paddle::framework::ir::ElementwiseAddActivationOneDNNPass);
REGISTER_PASS_CAPABILITY(el_add_act_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 0)
            .LE("relu", 0));
