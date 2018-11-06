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

std::pair<bool, Node*> ResidualConnectionMKLDNNFusePass::HasBias(
    const Node& op) const {
  auto bias_input_names = op.Op()->Inputs();
  auto bias_it = bias_input_names.find("Bias");

  if (bias_it != std::end(bias_input_names)) {
    bool has_bias = !bias_it->second.empty();

    if (has_bias) {
      auto bias_names = bias_it->second;
      auto bias_names_it =
          std::find_if(std::begin(op.inputs), std::end(op.inputs),
                       [&bias_names](Node* n) -> bool {
                         return n->Name() == bias_names[0];
                       });
      return std::make_pair(has_bias, *bias_names_it);
    }
  }

  return std::make_pair(false, nullptr);
}

graph_ptr ResidualConnectionMKLDNNFusePass::FuseConvAsX(
    const std::string& name_scope_, graph_ptr graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope_};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope_};
  elementwise_add_pattern(
      conv_output,
      pattern->NewNode(elementwise_add_pattern.elementwise_add_y_repr()));
  conv_output->AsIntermediate();

  auto get_node_from_conv = [](const patterns::Conv& conv_pattern,
                               const GraphPatternDetector::subgraph_t& subgraph)
      -> std::tuple<Node*, Node*, Node*, Node*> {
        GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

        return std::make_tuple(conv_op, conv_input, conv_filter, conv_output);
      };

  auto get_node_from_elementwise_add = [](
      const patterns::ElementwiseAdd& elementwise_add_pattern,
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

  auto handler =
      GenerateFuseHandler(conv_pattern, elementwise_add_pattern,
                          get_node_from_conv, get_node_from_elementwise_add);
  gpd(graph.get(), handler);

  return graph;
}

graph_ptr ResidualConnectionMKLDNNFusePass::FuseConvAsY(
    const std::string& name_scope_, graph_ptr graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope_};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope_};
  elementwise_add_pattern(
      pattern->NewNode(elementwise_add_pattern.elementwise_add_x_repr()),
      conv_output);
  conv_output->AsIntermediate();

  auto get_node_from_conv = [](const patterns::Conv& conv_pattern,
                               const GraphPatternDetector::subgraph_t& subgraph)
      -> std::tuple<Node*, Node*, Node*, Node*> {
        GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
        GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

        return std::make_tuple(conv_op, conv_input, conv_filter, conv_output);
      };

  auto get_node_from_elementwise_add = [](
      const patterns::ElementwiseAdd& elementwise_add_pattern,
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

  auto handler =
      GenerateFuseHandler(conv_pattern, elementwise_add_pattern,
                          get_node_from_conv, get_node_from_elementwise_add);
  gpd(graph.get(), handler);

  return graph;
}

graph_ptr ResidualConnectionMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  return FuseConvAsY(name_scope_, FuseConvAsX(name_scope_, std::move(graph)));
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::ResidualConnectionMKLDNNFusePass);
