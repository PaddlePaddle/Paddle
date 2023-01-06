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

#include "paddle/fluid/framework/ir/mkldnn/fc_act_mkldnn_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/utils/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseFCActOneDNNPass::ApplyImpl(Graph *graph) const {
  auto act_types = phi::funcs::GetSupportedActivations();

  for (auto act_type : act_types) FuseFCAct(graph, act_type);
}

void FuseFCActOneDNNPass::FuseFCAct(Graph *graph,
                                    const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("fc_" + act_type + "_mkldnn_fuse_pass", graph);

  GraphPatternDetector gpd;
  patterns::OperatorActivation fc_act_pattern(
      gpd.mutable_pattern(), "fc_" + act_type + "_mkldnn_fuse_pass");
  fc_act_pattern("fc", act_type);

  int found_fc_act_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse fc with activation op.";
    GET_IR_NODE_FROM_SUBGRAPH(fc, preceding_op, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, preceding_op_out, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, activation, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act_out, activation_out, fc_act_pattern);

    auto *fc_op = fc->Op();
    auto *act_op = act->Op();

    if (fc_op->HasAttr("use_mkldnn")) {
      PADDLE_ENFORCE(
          PADDLE_GET_CONST(bool, fc_op->GetAttr("use_mkldnn")),
          platform::errors::PreconditionNotMet(
              "The FC+Act fusion may happen only when oneDNN library "
              "is used."));
    }

    auto attr_map = phi::funcs::GetAttributeMap(act_type);
    for (const auto &attr : attr_map) {
      if (act_op->HasAttr(attr.first)) {
        fc_op->SetAttr(attr.second, act_op->GetAttr(attr.first));
      }
    }

    if (act_type == "gelu" && act_op->HasAttr("approximate")) {
      std::string gelu_act_type =
          PADDLE_GET_CONST(bool, act_op->GetAttr("approximate")) ? "gelu_tanh"
                                                                 : "gelu_erf";
      fc_op->SetAttr("fuse_activation", gelu_act_type);
    } else {
      fc_op->SetAttr("fuse_activation", act_type);
    }

    fc_op->SetAttr("use_mkldnn", true);
    fc_op->SetOutput("Out", {act_out->Name()});

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

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_act_mkldnn_fuse_pass,
              paddle::framework::ir::FuseFCActOneDNNPass);
REGISTER_PASS_CAPABILITY(fc_act_mkldnn_fuse_pass)
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
