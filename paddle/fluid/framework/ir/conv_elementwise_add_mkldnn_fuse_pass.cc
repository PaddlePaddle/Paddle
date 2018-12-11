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

#include "paddle/fluid/framework/ir/conv_elementwise_add_mkldnn_fuse_pass.h"
#include <functional>
#include <list>
#include <map>
#include <tuple>

#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {

// The function keeps the graph consistent by replacing
// a node 'from' in the set of inputs nodes
// of the visited node by a node 'to'.
void CorrectGraphEdges(Graph* graph, Node* from, Node* to) {
  for (auto& node : GraphTraits::DFS(*graph)) {
    auto from_in_inputs =
        std::find(std::begin(node.inputs), std::end(node.inputs), from);

    if (from_in_inputs != std::end(node.inputs)) {
      IR_NODE_LINK_TO(to, (&node));

      auto inputs = node.Op()->Inputs();

      using input_type = VariableNameMap::value_type;

      std::for_each(std::begin(inputs), std::end(inputs),
                    [from, to, &node](const input_type& i) -> void {
                      auto param_names = i.second;
                      auto pi = std::find(std::begin(param_names),
                                          std::end(param_names), from->Name());

                      if (pi != std::end(param_names)) {
                        node.Op()->SetInput(i.first, {to->Name()});
                      }
                    });
    }
  }
}

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

boost::optional<Node*> HasBias(const Node& op, const std::string& bias_name) {
  auto bias_input_names = op.Op()->Inputs();
  auto bias_it = bias_input_names.find(bias_name);

  if (bias_it != std::end(bias_input_names)) {
    bool has_bias = !bias_it->second.empty();

    if (has_bias) {
      auto bias_names = bias_it->second;
      auto bias_names_it =
          std::find_if(std::begin(op.inputs), std::end(op.inputs),
                       [&bias_names](Node* n) -> bool {
                         return n->Name() == bias_names[0];
                       });
      return *bias_names_it;
    }
  }

  return boost::none;
}

ResidualConnectionMKLDNNFusePass::IdentityFuseHandle::IdentityFuseHandle(
    const ResidualConnectionMKLDNNFusePass::CanFuseFunc& can_fuse_func,
    const ResidualConnectionMKLDNNFusePass::IdentityConvFunc&
        get_node_from_conv_op,
    const ResidualConnectionMKLDNNFusePass::IdentityElementwiseAddFunc&
        get_node_from_elementwise_add_op)
    : fusion_stats{std::make_shared<int>(0)},
      can_fuse_func{can_fuse_func},
      get_node_from_conv_op{get_node_from_conv_op},
      get_node_from_elementwise_add_op{get_node_from_elementwise_add_op} {}

void ResidualConnectionMKLDNNFusePass::IdentityFuseHandle::operator()(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
  Node* conv_op;
  Node* conv_input;
  Node* conv_filter;
  Node* conv_output;

  Node* elementwise_add_op;
  Node* elementwise_add_identity;
  Node* elementwise_add_out;

  std::tie(conv_op, conv_input, conv_filter, conv_output) =
      get_node_from_conv_op(subgraph);
  std::tie(elementwise_add_op, elementwise_add_identity, elementwise_add_out) =
      get_node_from_elementwise_add_op(subgraph);

  if (!can_fuse_func(conv_op, elementwise_add_op)) return;

  if (!IsReachable(graph, elementwise_add_identity, conv_output)) return;

  OpDesc op_desc;
  op_desc.SetType("conv2d");

  op_desc.SetInput("Input", {conv_input->Name()});
  op_desc.SetInput("Filter", {conv_filter->Name()});
  op_desc.SetInput("ResidualData", {elementwise_add_identity->Name()});
  op_desc.SetOutput("Output", {conv_output->Name()});

  auto conv_bias = HasBias(*conv_op, "Bias");

  if (conv_bias) {
    op_desc.SetInput("Bias", {(*conv_bias)->Name()});
  }

  for (const auto& attr : conv_op->Op()->GetAttrMap()) {
    op_desc.SetAttr(attr.first, attr.second);
  }

  op_desc.SetAttr("fuse_residual_connection", true);

  auto fused_conv_op = graph->CreateOpNode(&op_desc);

  IR_NODE_LINK_TO(conv_input, fused_conv_op);
  IR_NODE_LINK_TO(conv_filter, fused_conv_op);
  IR_NODE_LINK_TO(elementwise_add_identity, fused_conv_op);
  IR_NODE_LINK_TO(fused_conv_op, conv_output);

  if (conv_bias) {
    IR_NODE_LINK_TO((*conv_bias), fused_conv_op);
  }

  CorrectGraphEdges(graph, elementwise_add_out, conv_output);
  GraphSafeRemoveNodes(graph,
                       {elementwise_add_out, conv_op, elementwise_add_op});
  (*fusion_stats)++;
}

ResidualConnectionMKLDNNFusePass::ProjectionFuseHandle::ProjectionFuseHandle(
    const ResidualConnectionMKLDNNFusePass::CanFuseFunc& can_fuse_func,
    const ResidualConnectionMKLDNNFusePass::ProjectionConvFunc&
        get_node_from_conv_x_op,
    const ResidualConnectionMKLDNNFusePass::ProjectionConvFunc&
        get_node_from_conv_y_op,
    const ResidualConnectionMKLDNNFusePass::ProjectionElementwiseAddFunc&
        get_node_from_elementwise_add_op)
    : fusion_stats{std::make_shared<int>(0)},
      can_fuse_func{can_fuse_func},
      get_node_from_conv_x_op{get_node_from_conv_x_op},
      get_node_from_conv_y_op{get_node_from_conv_y_op},
      get_node_from_elementwise_add_op{get_node_from_elementwise_add_op} {}

void ResidualConnectionMKLDNNFusePass::ProjectionFuseHandle::operator()(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
  Node* conv_x_op;
  Node* conv_x_input;
  Node* conv_x_filter;
  Node* conv_x_output;

  Node* conv_y_op;
  Node* conv_y_input;
  Node* conv_y_filter;
  Node* conv_y_output;

  Node* elementwise_add_op;
  Node* elementwise_add_out;

  std::tie(conv_x_op, conv_x_input, conv_x_filter, conv_x_output) =
      get_node_from_conv_x_op(subgraph);
  std::tie(conv_y_op, conv_y_input, conv_y_filter, conv_y_output) =
      get_node_from_conv_y_op(subgraph);
  std::tie(elementwise_add_op, elementwise_add_out) =
      get_node_from_elementwise_add_op(subgraph);

  if (!can_fuse_func(conv_x_op, elementwise_add_op)) return;
  if (!can_fuse_func(conv_y_op, elementwise_add_op)) return;

  Node* projection_node;
  Node* residual_conv_op;
  Node* residual_conv_input;
  Node* residual_conv_filter;
  Node* residual_conv_output;

  if (IsReachable(graph, conv_x_input, conv_y_output)) {
    projection_node = conv_x_output;
    residual_conv_op = conv_y_op;
    residual_conv_input = conv_y_input;
    residual_conv_filter = conv_y_filter;
    residual_conv_output = conv_y_output;
  } else if (IsReachable(graph, conv_y_input, conv_x_output)) {
    projection_node = conv_y_output;
    residual_conv_op = conv_x_op;
    residual_conv_input = conv_x_input;
    residual_conv_filter = conv_x_filter;
    residual_conv_output = conv_x_output;
  } else {
    return;
  }

  OpDesc op_desc;
  op_desc.SetType("conv2d");

  op_desc.SetInput("Input", {residual_conv_input->Name()});
  op_desc.SetInput("Filter", {residual_conv_filter->Name()});
  op_desc.SetInput("ResidualData", {projection_node->Name()});
  op_desc.SetOutput("Output", {residual_conv_output->Name()});

  auto residual_conv_bias = HasBias(*residual_conv_op, "Bias");

  if (residual_conv_bias) {
    op_desc.SetInput("Bias", {(*residual_conv_bias)->Name()});
  }

  for (const auto& attr : residual_conv_op->Op()->GetAttrMap()) {
    op_desc.SetAttr(attr.first, attr.second);
  }

  op_desc.SetAttr("fuse_residual_connection", true);

  auto fused_conv_op = graph->CreateOpNode(&op_desc);

  IR_NODE_LINK_TO(residual_conv_input, fused_conv_op);
  IR_NODE_LINK_TO(residual_conv_filter, fused_conv_op);
  IR_NODE_LINK_TO(projection_node, fused_conv_op);
  IR_NODE_LINK_TO(fused_conv_op, residual_conv_output);

  if (residual_conv_bias) {
    IR_NODE_LINK_TO((*residual_conv_bias), fused_conv_op);
  }

  CorrectGraphEdges(graph, elementwise_add_out, residual_conv_output);
  GraphSafeRemoveNodes(
      graph, {elementwise_add_out, residual_conv_op, elementwise_add_op});
  (*fusion_stats)++;
}

std::tuple<Node*, Node*, Node*, Node*>
ResidualConnectionMKLDNNFusePass::GetNodesFromConv(
    const patterns::Conv& conv_pattern,
    const GraphPatternDetector::subgraph_t& subgraph) const {
  GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
  GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
  GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
  GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

  return std::make_tuple(conv_op, conv_input, conv_filter, conv_output);
}

GraphWithStats ResidualConnectionMKLDNNFusePass::FuseConvAsX(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  ir::Graph* graph;
  int stats;

  std::tie(graph, stats) = graph_with_stats;

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(
      conv_output,
      pattern->NewNode(elementwise_add_pattern.elementwise_add_y_repr()));
  conv_output->AsIntermediate();

  auto get_node_from_elementwise_add = [&elementwise_add_pattern](
      const GraphPatternDetector::subgraph_t& subgraph)
      -> std::tuple<Node*, Node*, Node*> {
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                                  elementwise_add_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_y, elementwise_add_y,
                                  elementwise_add_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                                  elementwise_add_pattern);

        return std::make_tuple(elementwise_add_op, elementwise_add_y,
                               elementwise_add_out);
      };

  return ExecuteHandleOnGraph<IdentityFuseHandle>(
      &gpd, graph_with_stats,
      [this, &conv_pattern](const GraphPatternDetector::subgraph_t& subgraph) {
        return GetNodesFromConv(conv_pattern, subgraph);
      },
      get_node_from_elementwise_add);
}

GraphWithStats ResidualConnectionMKLDNNFusePass::FuseConvAsY(
    const std::string& name_scope,
    const GraphWithStats& graph_with_stats) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(
      pattern->NewNode(elementwise_add_pattern.elementwise_add_x_repr()),
      conv_output);
  conv_output->AsIntermediate();

  auto get_node_from_elementwise_add = [&elementwise_add_pattern](
      const GraphPatternDetector::subgraph_t& subgraph)
      -> std::tuple<Node*, Node*, Node*> {
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                                  elementwise_add_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_x, elementwise_add_x,
                                  elementwise_add_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                                  elementwise_add_pattern);

        return std::make_tuple(elementwise_add_op, elementwise_add_x,
                               elementwise_add_out);
      };

  return ExecuteHandleOnGraph<IdentityFuseHandle>(
      &gpd, graph_with_stats,
      [this, &conv_pattern](const GraphPatternDetector::subgraph_t& subgraph) {
        return GetNodesFromConv(conv_pattern, subgraph);
      },
      get_node_from_elementwise_add);
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

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope};
  elementwise_add_pattern(conv_x_output, conv_y_output);
  conv_x_output->AsIntermediate();
  conv_y_output->AsIntermediate();

  auto get_node_from_elementwise_add = [&elementwise_add_pattern](
      const GraphPatternDetector::subgraph_t& subgraph)
      -> std::tuple<Node*, Node*> {
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                                  elementwise_add_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                                  elementwise_add_pattern);

        return std::make_tuple(elementwise_add_op, elementwise_add_out);
      };

  return ExecuteHandleOnGraph<ProjectionFuseHandle>(
      &gpd, graph_with_stats,
      [this,
       &conv_x_pattern](const GraphPatternDetector::subgraph_t& subgraph) {
        return GetNodesFromConv(conv_x_pattern, subgraph);
      },
      [this,
       &conv_y_pattern](const GraphPatternDetector::subgraph_t& subgraph) {
        return GetNodesFromConv(conv_y_pattern, subgraph);
      },
      get_node_from_elementwise_add);
}

graph_ptr ResidualConnectionMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init(name_scope_, graph.get());
  auto fused_graph_with_stats = FuseConvAsY(
      name_scope_,
      FuseConvAsX(
          name_scope_,
          FuseProjectionConv(name_scope_, std::make_pair(graph.get(), 0))));

  std::cout << "Fused graph " << fused_graph_with_stats.second << std::endl;
  AddStatis(fused_graph_with_stats.second);
  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::ResidualConnectionMKLDNNFusePass);
