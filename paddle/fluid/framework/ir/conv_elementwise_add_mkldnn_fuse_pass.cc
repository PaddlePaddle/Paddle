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
#include <utility>

#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {
namespace {

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
}  // namespace
using graph_ptr = std::unique_ptr<ir::Graph>;

graph_ptr ConvElementwiseAddMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init(name_scope_, graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, name_scope_};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope_};
  elementwise_add_pattern(conv_output);

  conv_output->AsIntermediate();

  auto conv_op_has_bias = [](const Node& conv_op) -> std::pair<bool, Node*> {
    auto bias_input_names = conv_op.Op()->Inputs();
    auto bias_it = bias_input_names.find("Bias");

    if (bias_it != std::end(bias_input_names)) {
      bool has_bias = !bias_it->second.empty();

      if (has_bias) {
        auto conv_bias_names = bias_it->second;
        auto conv_bias_names_it =
            std::find_if(std::begin(conv_op.inputs), std::end(conv_op.inputs),
                         [&conv_bias_names](Node* n) -> bool {
                           return n->Name() == conv_bias_names[0];
                         });
        return std::make_pair(has_bias, *conv_bias_names_it);
      }
    }

    return std::make_pair(false, nullptr);
  };

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_x, elementwise_add_x,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    if (FindFuseOption(*conv_op, *elementwise_add_op) != FUSE_MKLDNN) return;

    OpDesc op_desc;
    op_desc.SetType("conv2d");

    op_desc.SetInput("Input", {conv_input->Name()});
    op_desc.SetInput("Filter", {conv_filter->Name()});
    op_desc.SetInput("ResidualData", {elementwise_add_x->Name()});
    op_desc.SetOutput("Output", {conv_output->Name()});

    bool has_bias;
    Node* conv_bias;

    std::tie(has_bias, conv_bias) = conv_op_has_bias(*conv_op);

    if (has_bias) {
      op_desc.SetInput("Bias", {conv_bias->Name()});
    }

    for (const auto& attr : conv_op->Op()->GetAttrMap()) {
      op_desc.SetAttr(attr.first, attr.second);
    }

    op_desc.SetAttr("fuse_residual_connection", true);

    auto fused_conv_op = g->CreateOpNode(&op_desc);

    IR_NODE_LINK_TO(conv_input, fused_conv_op);
    IR_NODE_LINK_TO(conv_filter, fused_conv_op);
    IR_NODE_LINK_TO(elementwise_add_x, fused_conv_op);
    IR_NODE_LINK_TO(fused_conv_op, conv_output);

    if (has_bias) {
      IR_NODE_LINK_TO(conv_bias, fused_conv_op);
    }

    CorrectGraphEdges(g, elementwise_add_out, conv_output);
    GraphSafeRemoveNodes(g, {elementwise_add_out, conv_op, elementwise_add_op});
  };

  gpd(graph.get(), handler);

  return graph;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_mkldnn_fuse_pass,
              paddle::framework::ir::ConvElementwiseAddMKLDNNFusePass);
