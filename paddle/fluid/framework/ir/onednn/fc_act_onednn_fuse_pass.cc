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

#include "paddle/fluid/framework/ir/onednn/fc_act_onednn_fuse_pass.h"

#include "paddle/fluid/framework/ir/onednn/activation_onednn_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle::framework::ir {

using string::PrettyLogDetail;

void FuseFCActOneDNNPass::ApplyImpl(Graph *graph) const {
  auto act_types = GetSupportedActivations();

  for (auto const &act_type : act_types) FuseFCAct(graph, act_type);
}

void FuseFCActOneDNNPass::FuseFCAct(Graph *graph,
                                    const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("fc_" + act_type + "_onednn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation fc_act_pattern(
      gpd.mutable_pattern(), "fc_" + act_type + "_onednn_fuse_pass");
  fc_act_pattern("fc", act_type);

  int found_fc_act_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse fc with activation op.";
    GET_IR_NODE_FROM_SUBGRAPH(fc, preceding_op, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, preceding_op_out, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, activation, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, activation_out, fc_act_pattern);

    SetActivationAttrs(fc->Op(), act->Op(), act_type);
    fc->Op()->SetOutput("Out", {act_out->Name()});

    IR_OP_VAR_LINK(fc, act_out);
    GraphSafeRemoveNodes(g, {act, fc_out});
    found_fc_act_count++;
  };

  gpd(graph, handler);
  AddStatis(found_fc_act_count);
  if ((!Has("disable_logs") || !Get<bool>("disable_logs")) &&
      found_fc_act_count > 0)
    PrettyLogDetail(
        "---    fused %d fc with %s activation", found_fc_act_count, act_type);
}

}  // namespace paddle::framework::ir

REGISTER_PASS(fc_act_onednn_fuse_pass,
              paddle::framework::ir::FuseFCActOneDNNPass);
REGISTER_PASS_CAPABILITY(fc_act_onednn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("fc", 0)
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
