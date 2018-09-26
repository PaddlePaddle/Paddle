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

#include "paddle/fluid/framework/ir/graph_traits.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

template <typename IT, typename FindFunc, typename ReplaceFunc>
static void ReplaceAllOccurances(IT s, IT e, FindFunc f, ReplaceFunc r) {
  if (s == e) return;

  auto it = std::find_if(s, e, f);

  if (it != e) {
    r(*it);
  }

  it++;
  ReplaceAllOccurances(it, e, f, r);
}

static void CorrectGraphEdges(Graph* graph, Node* from, Node* to) {
  for (auto& node : GraphTraits::DFS(*graph)) {
    auto same = std::find_if(std::begin(node.inputs), std::end(node.inputs),
                             [from](Node* n) { return n == from; });

    if (same != std::end(node.inputs)) {
      IR_NODE_LINK_TO(to, (&node));

      auto inputs = node.Op()->Inputs();

      using input_type = VariableNameMap::value_type;

      ReplaceAllOccurances(
          std::begin(inputs), std::end(inputs),
          [from](const input_type& i) -> bool {
            auto params = i.second;
            auto pi =
                std::find_if(std::begin(params), std::end(params),
                             std::bind(std::equal_to<std::string>(),
                                       from->Name(), std::placeholders::_1));
            return pi != std::end(params);
          },
          [to, &node](const input_type& i) {
            node.Op()->SetInput(i.first, {to->Name()});
          });
    }
  }
}
}  // namespace patterns
using graph_ptr = std::unique_ptr<ir::Graph>;

graph_ptr ConvElementwiseAddMKLDNNFusePass::ApplyImpl(graph_ptr graph) const {
  FusePassBase::Init("conv_elementwise_add_mkldnn_fuse_pass", graph.get());

  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();

  patterns::Conv conv_pattern{pattern, "skip_connections_fusion"};
  auto conv_output = conv_pattern();

  patterns::ElementwiseAdd elementwise_add_pattern{pattern,
                                                   "skip_connections_fusion"};
  elementwise_add_pattern(conv_output);

  conv_output->AsIntermediate();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_bias, conv_bias, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_x, elementwise_add_x,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    OpDesc op_desc;
    op_desc.SetType("conv2d");

    op_desc.SetInput("Input", {conv_input->Name()});
    op_desc.SetInput("Bias", {conv_bias->Name()});
    op_desc.SetInput("Filter", {conv_filter->Name()});
    op_desc.SetInput("ResidualData", {elementwise_add_x->Name()});
    op_desc.SetOutput("Output", {conv_output->Name()});

    op_desc.SetAttr("use_mkldnn", true);
    op_desc.SetAttr("fuse_residual_connection", true);

    auto fused_conv_op = g->CreateOpNode(&op_desc);

    IR_NODE_LINK_TO(conv_input, fused_conv_op);
    IR_NODE_LINK_TO(conv_bias, fused_conv_op);
    IR_NODE_LINK_TO(conv_filter, fused_conv_op);
    IR_NODE_LINK_TO(elementwise_add_x, fused_conv_op);
    IR_NODE_LINK_TO(fused_conv_op, conv_output);

    patterns::CorrectGraphEdges(g, elementwise_add_out, conv_output);
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
