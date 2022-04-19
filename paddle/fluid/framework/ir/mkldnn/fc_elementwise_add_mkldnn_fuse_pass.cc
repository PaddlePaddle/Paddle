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

#include "paddle/fluid/framework/ir/mkldnn/fc_elementwise_add_mkldnn_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

FCResidualConnectionMKLDNNFusePass::FCResidualConnectionMKLDNNFusePass() {
  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsIntIn({-1, 0, 1})
      .End();
}

GraphWithStats FCResidualConnectionMKLDNNFusePass::FuseFC(
    const std::string& name_scope, const GraphWithStats& graph_with_stats,
    bool fc_as_x) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::FCMKLDNN fc_pattern{pattern, name_scope};
  bool fc_has_bias = true;
  auto fc_output = fc_pattern(
      gpd.mutable_pattern()->NewNode("fc")->AsInput()->assert_is_op_input(
          "fc", "Input"),
      fc_has_bias);

  patterns::ResidualElementwise elementwise_pattern{pattern, name_scope,
                                                    fc_as_x};
  elementwise_pattern(
      fc_output, pattern->NewNode(elementwise_pattern.residual_data_repr()),
      "elementwise_add", fc_as_x);
  fc_output->AsIntermediate();

  int found_fc_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(residual_data, residual_data,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (FindFuseOption(*fc_op, *elementwise_op) != FUSE_MKLDNN) return;
    if (!IsReachable(g, residual_data, fc_output)) return;
    if (HasFusedActivation(fc_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "op compat for fc_elementwise_add_mkldnn_fuse_pass failed.";
      return;
    }

    fc_op->Op()->SetOutput("ResidualData", {residual_data->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_op});

    IR_NODE_LINK_TO(residual_data, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_out);

    found_fc_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    std::string fusionMode = fc_as_x ? "x" : "y";
    msg_ss << "---    Fused " << found_fc_count << " fc (as " << fusionMode
           << ") + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_fc_count + graph_with_stats.second);
}

void FCResidualConnectionMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto graph_with_stats = FuseFC(name_scope_, std::make_pair(graph, 0), true);
  graph_with_stats = FuseFC(name_scope_, graph_with_stats, false);

  AddStatis(graph_with_stats.second);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::FCResidualConnectionMKLDNNFusePass);
REGISTER_PASS_CAPABILITY(fc_elementwise_add_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("fc", 0)
            .LE("elementwise_add", 1));
