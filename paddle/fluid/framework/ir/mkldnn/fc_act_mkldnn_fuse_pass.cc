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

#include "paddle/fluid/framework/ir/mkldnn/fc_act_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseFCActOneDNNPass::ApplyImpl(Graph *graph) const {
  std::vector<std::string> act_types = {"gelu", "tanh", "sigmoid", "mish",
                                        "hard_swish"};

  for (std::string act_type : act_types) FuseFCAct(graph, act_type);
}

void FuseFCActOneDNNPass::FuseFCAct(Graph *graph,
                                    const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("fc_act", graph);

  GraphPatternDetector gpd;
  patterns::FCActOneDNN fc_act_pattern(gpd.mutable_pattern(), "fc_act");
  fc_act_pattern(act_type);

  int found_fc_act_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse fc with activation op.";
    // FC output
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, fc_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, fc_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, fc_act_pattern);

    auto *fc_op = fc->Op();
    auto *act_op = act->Op();

    if (fc_op->HasAttr("use_mkldnn")) {
      PADDLE_ENFORCE(
          BOOST_GET_CONST(bool, fc_op->GetAttr("use_mkldnn")),
          platform::errors::PreconditionNotMet(
              "The FC+Act fusion may happen only when oneDNN library "
              "is used."));
    }

    if (act_type == "gelu" && act_op->HasAttr("approximate")) {
      bool approximate = BOOST_GET_CONST(bool, act_op->GetAttr("approximate"));
      std::string type = approximate ? "_tanh" : "_erf";
      fc_op->SetAttr("activation_type", act_type + type);
    } else {
      fc_op->SetAttr("activation_type", act_type);
    }
    fc_op->SetAttr("use_mkldnn", true);

    fc_op->SetOutput("Out", {act_out->Name()});

    IR_OP_VAR_LINK(fc, act_out);
    GraphSafeRemoveNodes(g, {act, fc_out});
    found_fc_act_count++;
  };

  gpd(graph, handler);
  AddStatis(found_fc_act_count);
  if (!Has("disable_logs") || !Get<bool>("disable_logs"))
    PrettyLogDetail("---    fused %d fc with %s activation", found_fc_act_count,
                    act_type);
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
            .LE("gelu", 0)
            .LE("sigmoid", 0)
            .LE("mish", 1)
            .LE("hard_swish", 0)
            .LE("tanh", 0));
