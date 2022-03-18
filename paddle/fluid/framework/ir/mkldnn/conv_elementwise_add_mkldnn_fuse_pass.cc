// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_elementwise_add_mkldnn_fuse_pass.h"

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <tuple>

#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

bool IsReachable(ir::Graph* graph, Node* from, Node* to) {
  auto find_node = [](ir::Graph* graph, const Node* node) -> Node* {
    for (auto n : graph->Nodes()) {
      if (n == node) {
        return n;
      }
    }

    return nullptr;
  };

  if (from == to) {
    return true;
  }

  std::map<Node*, bool> visited;

  for (auto& node : GraphTraits::DFS(*graph)) {
    visited[&node] = false;
  }

  visited[from] = true;

  std::list<Node*> queue;
  queue.push_back(from);

  while (!queue.empty()) {
    auto cur = find_node(graph, queue.front());
    queue.pop_front();

    if (!cur) return false;

    for (auto n : cur->outputs) {
      if (n == to) {
        return true;
      }

      if (!visited[n]) {
        visited[n] = true;
        queue.push_back(n);
      }
    }
  }
  return false;
}

template <typename T>
paddle::optional<T> HasAttribute(const Node& op, const std::string& attr) {
  if (op.Op()->HasAttr(attr))
    return BOOST_GET_CONST(T, op.Op()->GetAttr(attr));
  else
    return paddle::none;
}

ResidualConnectionMKLDNNFusePass::ResidualConnectionMKLDNNFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
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
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NHWC", "NCHW", "AnyLayout"})
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

GraphWithStats ResidualConnectionMKLDNNFusePass::FuseConvAsX(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope};
  auto conv_output = conv_pattern();

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(
      conv_output, pattern->NewNode(elementwise_pattern.elementwise_y_repr()),
      "elementwise_add");
  conv_output->AsIntermediate();

  int found_conv_as_x_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_identity, elementwise_y,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (FindFuseOption(*conv_op, *elementwise_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_identity, conv_output)) return;

    if (HasFusedActivation(conv_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "conv_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    conv_op->Op()->SetInput("ResidualData", {elementwise_identity->Name()});
    conv_op->Op()->SetOutput("Output", {elementwise_out->Name()});
    conv_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {conv_output, elementwise_op});

    IR_NODE_LINK_TO(elementwise_identity, conv_op);
    IR_NODE_LINK_TO(conv_op, elementwise_out);

    found_conv_as_x_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_conv_as_x_count
           << " conv (as x) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_conv_as_x_count + graph_with_stats.second);
}

GraphWithStats ResidualConnectionMKLDNNFusePass::FuseConvAsY(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope};
  auto conv_output = conv_pattern();

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(
      pattern->NewNode(elementwise_pattern.elementwise_x_repr()), conv_output,
      "elementwise_add");
  conv_output->AsIntermediate();

  int found_conv_as_y_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_x, elementwise_x,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (FindFuseOption(*conv_op, *elementwise_op) != FUSE_MKLDNN) return;

    if (!IsReachable(g, elementwise_x, conv_output)) return;

    if (HasFusedActivation(conv_op)) return;

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "conv_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    conv_op->Op()->SetInput("ResidualData", {elementwise_x->Name()});
    conv_op->Op()->SetOutput("Output", {elementwise_out->Name()});
    conv_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {conv_output, elementwise_op});

    IR_NODE_LINK_TO(elementwise_x, conv_op);
    IR_NODE_LINK_TO(conv_op, elementwise_out);

    found_conv_as_y_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_conv_as_y_count
           << " conv (as y) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_conv_as_y_count + graph_with_stats.second);
}

GraphWithStats ResidualConnectionMKLDNNFusePass::FuseProjectionConv(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_x_pattern{pattern, name_scope};
  auto conv_x_output = conv_x_pattern();

  patterns::Conv conv_y_pattern{pattern, name_scope};
  auto conv_y_output = conv_y_pattern();

  patterns::Elementwise elementwise_pattern{pattern, name_scope};
  elementwise_pattern(conv_x_output, conv_y_output, "elementwise_add");
  conv_x_output->AsIntermediate();
  conv_y_output->AsIntermediate();

  int found_projection_conv_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(conv_x_op, conv_op, conv_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_x_input, conv_input, conv_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_x_filter, conv_filter, conv_x_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_x_output, conv_output, conv_x_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(conv_y_op, conv_op, conv_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_y_input, conv_input, conv_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_y_filter, conv_filter, conv_y_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_y_output, conv_output, conv_y_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_op, elementwise_op,
                              elementwise_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out,
                              elementwise_pattern);

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "conv_elementwise_add_mkldnn_fuse_pass in op compat failed.";
      return;
    }

    if (FindFuseOption(*conv_x_op, *elementwise_op) != FUSE_MKLDNN) return;
    if (FindFuseOption(*conv_y_op, *elementwise_op) != FUSE_MKLDNN) return;

    Node* projection_node;
    Node* residual_conv_op;
    Node* residual_conv_output;
    if (IsReachable(g, conv_x_input, conv_y_output)) {
      projection_node = conv_x_output;
      residual_conv_op = conv_y_op;
      residual_conv_output = conv_y_output;
    } else if (IsReachable(g, conv_y_input, conv_x_output)) {
      projection_node = conv_y_output;
      residual_conv_op = conv_x_op;
      residual_conv_output = conv_x_output;
    } else {
      return;
    }

    if (HasFusedActivation(residual_conv_op)) return;

    residual_conv_op->Op()->SetInput("ResidualData", {projection_node->Name()});
    residual_conv_op->Op()->SetOutput("Output", {elementwise_out->Name()});

    residual_conv_op->Op()->SetAttr("fuse_residual_connection", true);

    GraphSafeRemoveNodes(g, {residual_conv_output, elementwise_op});

    IR_NODE_LINK_TO(projection_node, residual_conv_op);
    IR_NODE_LINK_TO(residual_conv_op, elementwise_out);

    found_projection_conv_count++;
  };

  gpd(graph_with_stats.first, handler);
  if (!Has("disable_logs") || !Get<bool>("disable_logs")) {
    std::stringstream msg_ss;
    msg_ss << "---    Fused " << found_projection_conv_count
           << " projection conv (as y) + elementwise_add patterns";
    paddle::string::PrettyLogDetail(msg_ss.str().c_str());
  }

  return std::make_pair(graph_with_stats.first,
                        found_projection_conv_count + graph_with_stats.second);
}

void ResidualConnectionMKLDNNFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto graph_with_stats =
      FuseProjectionConv(name_scope_, std::make_pair(graph, 0));
  graph_with_stats = FuseConvAsX(name_scope_, graph_with_stats);
  graph_with_stats = FuseConvAsY(name_scope_, graph_with_stats);

  AddStatis(graph_with_stats.second);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::ResidualConnectionMKLDNNFusePass);
REGISTER_PASS_CAPABILITY(conv_elementwise_add_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1));
