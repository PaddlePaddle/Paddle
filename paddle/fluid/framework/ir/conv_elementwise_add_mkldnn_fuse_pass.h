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

#pragma once

#include <string>
#include <utility>
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

using graph_ptr = std::unique_ptr<ir::Graph>;

void CorrectGraphEdges(Graph* graph, Node* from, Node* to);
bool IsReachable(ir::Graph* graph, Node* from, Node* to);

using handler_func = std::function<void(
    const GraphPatternDetector::subgraph_t& subgraph, Graph* g)>;

class ResidualConnectionMKLDNNFusePass : public FusePassBase {
 private:
  graph_ptr FuseConvAsX(const std::string& name_scope_, graph_ptr graph) const;
  graph_ptr FuseConvAsY(const std::string& name_scope_, graph_ptr graph) const;

  std::pair<bool, Node*> HasBias(const Node& op) const;

  template <typename CONV_FUNC, typename ELEMENTWISE_ADD_FUNC,
            typename HANDLER_FUNC = handler_func>
  HANDLER_FUNC GenerateFuseHandler(
      const patterns::Conv& conv_pattern,
      const patterns::ElementwiseAdd& elementwise_add_pattern,
      CONV_FUNC get_node_from_conv_op,
      ELEMENTWISE_ADD_FUNC get_node_from_elementwise_add_op) const;

 public:
  virtual ~ResidualConnectionMKLDNNFusePass() {}

 protected:
  std::unique_ptr<ir::Graph> ApplyImpl(graph_ptr graph) const;

  const std::string name_scope_{"residual_connection_fuse_pass"};
};

template <typename CONV_FUNC, typename ELEMENTWISE_ADD_FUNC,
          typename HANDLER_FUNC>
HANDLER_FUNC ResidualConnectionMKLDNNFusePass::GenerateFuseHandler(
    const patterns::Conv& conv_pattern,
    const patterns::ElementwiseAdd& elementwise_add_pattern,
    CONV_FUNC get_node_from_conv_op,
    ELEMENTWISE_ADD_FUNC get_node_from_elementwise_add_op) const {
  return [&](const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    Node* conv_op;
    Node* conv_input;
    Node* conv_filter;
    Node* conv_output;

    Node* elementwise_add_op;
    Node* elementwise_add_identity;
    Node* elementwise_add_out;

    std::tie(conv_op, conv_input, conv_filter, conv_output) =
        get_node_from_conv_op(conv_pattern, subgraph);
    std::tie(elementwise_add_op, elementwise_add_identity,
             elementwise_add_out) =
        get_node_from_elementwise_add_op(elementwise_add_pattern, subgraph);

    if (this->FindFuseOption(*conv_op, *elementwise_add_op) != FUSE_MKLDNN)
      return;

    if (!IsReachable(graph, elementwise_add_identity, conv_output)) return;

    OpDesc op_desc;
    op_desc.SetType("conv2d");

    op_desc.SetInput("Input", {conv_input->Name()});
    op_desc.SetInput("Filter", {conv_filter->Name()});
    op_desc.SetInput("ResidualData", {elementwise_add_identity->Name()});
    op_desc.SetOutput("Output", {conv_output->Name()});

    bool has_bias;
    Node* conv_bias;

    std::tie(has_bias, conv_bias) = this->HasBias(*conv_op);

    if (has_bias) {
      op_desc.SetInput("Bias", {conv_bias->Name()});
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

    if (has_bias) {
      IR_NODE_LINK_TO(conv_bias, fused_conv_op);
    }

    CorrectGraphEdges(graph, elementwise_add_out, conv_output);
    GraphSafeRemoveNodes(graph,
                         {elementwise_add_out, conv_op, elementwise_add_op});
  };
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
