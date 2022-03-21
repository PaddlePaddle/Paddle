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

#include <functional>
#include <memory>
#include <tuple>

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
      .AddAttr("fuse_residual_connection")
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("Scale_in_eltwise")
      .IsOptional()
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

GraphWithStats FCResidualConnectionMKLDNNFusePass::FuseFCAsX(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::FCMKLDNN fc_pattern{pattern, name_scope};
  auto fc_output = fc_pattern(
      gpd.mutable_pattern()->NewNode("fc")->AsInput()->assert_is_op_input(
          "fc", "Input"),
      true);

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(
      fc_output, pattern->NewNode(elementwise_pattern.elementwise_y_repr()),
      "elementwise_add");
  fc_output->AsIntermediate();

  int found_fc_as_x_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_y, elementwise_y,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (FindFuseOption(*fc_op, *elementwise_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_y, fc_output)) return;

    if (HasFusedActivation(fc_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    fc_op->Op()->SetInput("ResidualData", {elementwise_y->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_op});

    IR_NODE_LINK_TO(elementwise_y, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_out);

    found_fc_as_x_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_fc_as_x_count
           << " fc (as x) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_fc_as_x_count + graph_with_stats.second);
}

GraphWithStats FCResidualConnectionMKLDNNFusePass::FuseFCAsY(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::FCMKLDNN fc_pattern{pattern, name_scope};
  auto fc_output = fc_pattern(
      gpd.mutable_pattern()->NewNode("fc")->AsInput()->assert_is_op_input(
          "fc", "Input"),
      true);

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(
      pattern->NewNode(elementwise_pattern.elementwise_x_repr()), fc_output,
      "elementwise_add");
  fc_output->AsIntermediate();

  int found_fc_as_y_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_x, elementwise_x,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (FindFuseOption(*fc_op, *elementwise_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_x, fc_output)) return;

    if (HasFusedActivation(fc_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    fc_op->Op()->SetInput("ResidualData", {elementwise_x->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_op});

    IR_NODE_LINK_TO(elementwise_x, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_out);

    found_fc_as_y_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_fc_as_y_count
           << " fc (as y) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_fc_as_y_count + graph_with_stats.second);
}

GraphWithStats FCResidualConnectionMKLDNNFusePass::FuseProjectionFC(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::FCMKLDNN fc_x_pattern{pattern, name_scope};

  auto fc_x_output = fc_x_pattern(
      gpd.mutable_pattern()->NewNode("fc_x")->AsInput()->assert_is_op_input(
          "fc", "Input"),
      true);

  patterns::FCMKLDNN fc_y_pattern{pattern, name_scope};
  auto fc_y_output = fc_y_pattern(
      gpd.mutable_pattern()->NewNode("fc_y")->AsInput()->assert_is_op_input(
          "fc", "Input"),
      true);

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(fc_x_output, fc_y_output, "elementwise_add");
  fc_x_output->AsIntermediate();
  fc_y_output->AsIntermediate();

  int found_projection_fc_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_op, fc, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_input, input, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_weights, weights, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_output, output, fc_x_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fc_y_op, fc, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_input, input, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_weights, weights, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_output, output, fc_y_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (HasFusedActivation(fc_x_op) || HasFusedActivation(fc_y_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    if (FindFuseOption(*fc_x_op, *elementwise_op) != FUSE_MKLDNN) return;
    if (FindFuseOption(*fc_y_op, *elementwise_op) != FUSE_MKLDNN) return;

    Node* projection_node;
    Node* residual_fc_op;
    Node* residual_fc_output;
    if (IsReachable(g, fc_x_input, fc_y_output)) {
      projection_node = fc_x_output;
      residual_fc_op = fc_y_op;
      residual_fc_output = fc_y_output;
    } else if (IsReachable(g, fc_y_input, fc_x_output)) {
      projection_node = fc_y_output;
      residual_fc_op = fc_x_op;
      residual_fc_output = fc_x_output;
    } else {
      return;
    }

    residual_fc_op->Op()->SetInput("ResidualData", {projection_node->Name()});
    residual_fc_op->Op()->SetOutput("Out", {elementwise_out->Name()});

    residual_fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {residual_fc_output, elementwise_op});

    IR_NODE_LINK_TO(projection_node, residual_fc_op);
    IR_NODE_LINK_TO(residual_fc_op, elementwise_out);

    found_projection_fc_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_projection_fc_count
           << " projection fc (as y) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_projection_fc_count + graph_with_stats.second);
}

void FCResidualConnectionMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto graph_with_stats =
      FuseProjectionFC(name_scope_, std::make_pair(graph, 0));
  graph_with_stats = FuseFCAsX(name_scope_, graph_with_stats);
  graph_with_stats = FuseFCAsY(name_scope_, graph_with_stats);

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
