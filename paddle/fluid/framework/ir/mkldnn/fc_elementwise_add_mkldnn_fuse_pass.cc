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
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"relu", ""})
      .IsOptional()
      .End()
      .AddAttr("padding_weights")
      .IsOptional()
      .End()
      .AddAttr("in_num_col_dims")
      .IsOptional()
      .End()
      .AddAttr("use_mkldnn")
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

  patterns::FCResidual fc_pattern{pattern, name_scope};
  auto fc_output = fc_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(
      fc_output,
      pattern->NewNode(elementwise_add_pattern.elementwise_add_y_repr()));
  fc_output->AsIntermediate();

  int found_fc_as_x_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc_op, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, fc_input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, fc_weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias, fc_bias, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, fc_output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_y, elementwise_add_y,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    if (FindFuseOption(*fc_op, *elementwise_add_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_add_y, fc_output)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    fc_op->Op()->SetInput("ResidualData", {elementwise_add_y->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_add_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_add_op});

    IR_NODE_LINK_TO(elementwise_add_y, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_add_out);

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

  patterns::FCResidual fc_pattern{pattern, name_scope};
  auto fc_output = fc_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(
      pattern->NewNode(elementwise_add_pattern.elementwise_add_x_repr()),
      fc_output);
  fc_output->AsIntermediate();

  int found_fc_as_y_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_op, fc_op, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_input, fc_input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_weights, fc_weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias, fc_bias, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_output, fc_output, fc_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_x, elementwise_add_x,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    if (FindFuseOption(*fc_op, *elementwise_add_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_add_x, fc_output)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    fc_op->Op()->SetInput("ResidualData", {elementwise_add_x->Name()});
    fc_op->Op()->SetOutput("Out", {elementwise_add_out->Name()});
    fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {fc_output, elementwise_add_op});

    IR_NODE_LINK_TO(elementwise_add_x, fc_op);
    IR_NODE_LINK_TO(fc_op, elementwise_add_out);

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

  patterns::FCResidual fc_x_pattern{pattern, name_scope};
  auto fc_x_output = fc_x_pattern();

  patterns::FCResidual fc_y_pattern{pattern, name_scope};
  auto fc_y_output = fc_y_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(fc_x_output, fc_y_output);
  fc_x_output->AsIntermediate();
  fc_y_output->AsIntermediate();

  int found_projection_fc_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_op, fc_op, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_input, fc_input, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_weights, fc_weights, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_bias, fc_bias, fc_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_x_output, fc_output, fc_x_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(fc_y_op, fc_op, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_input, fc_input, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_weights, fc_weights, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_bias, fc_bias, fc_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_y_output, fc_output, fc_y_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "fc_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    if (FindFuseOption(*fc_x_op, *elementwise_add_op) != FUSE_MKLDNN) return;
    if (FindFuseOption(*fc_y_op, *elementwise_add_op) != FUSE_MKLDNN) return;

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
    residual_fc_op->Op()->SetOutput("Out", {elementwise_add_out->Name()});

    residual_fc_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {residual_fc_output, elementwise_add_op});

    IR_NODE_LINK_TO(projection_node, residual_fc_op);
    IR_NODE_LINK_TO(residual_fc_op, elementwise_add_out);

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