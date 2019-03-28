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

#include <string>
#include "paddle/fluid/framework/ir/conv_elementwise_add2_act_fuse_pass.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                      \
  GET_IR_NODE(conv_op);                \
  GET_IR_NODE(conv_out);               \
  GET_IR_NODE(conv_filter);            \
  GET_IR_NODE(elementwise_add_op);     \
  GET_IR_NODE(elementwise_add_in_y);   \
  GET_IR_NODE(elementwise_add_out);    \
  GET_IR_NODE(elementwise_add_op_1);   \
  GET_IR_NODE(elementwise_add_in_y_1); \
  GET_IR_NODE(elementwise_add_out_1);  \
  GET_IR_NODE(act_op);                 \
  GET_IR_NODE(act_out);

// Inherient the basic infomation from `base_desc`, and modify some fields.
framework::proto::OpDesc PrepareOpDesc(
    const framework::proto::OpDesc& base_desc, const std::string& bias,
    const std::string& bias1, const std::string& activation,
    const std::string& output) {
  auto proto = base_desc;
  framework::OpDesc desc(proto, nullptr);
  desc.SetInput("Bias", {bias});
  desc.SetInput("ResidualData", {bias1});
  desc.SetAttr("activation", activation);
  desc.SetOutput("Output", {output});
  desc.SetAttr("is_test", true);
  desc.SetAttr("use_cudnn", false);

  return *desc.Proto();
}

void ConvElementwiseAddActFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "conv_elementwise_add_act_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()->NewNode("x")->AsInput()->assert_is_op_input(
      "conv2d", "Input");

  patterns::ConvElementwiseaddAct pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    auto base_op_desc = *conv_op->Op()->Proto();
    std::string bias_name = elementwise_add_in_y->Name();
    std::string bias1_name = elementwise_add_in_y_1->Name();
    std::string act_op_type = act_op->Op()->Type();
    std::string act_op_out = act_out->Name();

    auto new_op_proto = PrepareOpDesc(base_op_desc, bias_name, bias1_name,
                                      act_op_type, act_op_out);
    framework::OpDesc new_op_desc(new_op_proto, nullptr);

    // Create a new node for the fused op.
    auto new_conv_op = graph->CreateOpNode(&new_op_desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE(subgraph.count(x));
    auto* conv_in_node = subgraph.at(x);

    IR_NODE_LINK_TO(conv_in_node, new_conv_op);            // Input
    IR_NODE_LINK_TO(conv_filter, new_conv_op);             // Filter
    IR_NODE_LINK_TO(elementwise_add_in_y, new_conv_op);    // Bias
    IR_NODE_LINK_TO(elementwise_add_in_y_1, new_conv_op);  // ResidualData
    IR_NODE_LINK_TO(new_conv_op, act_out);                 // Output

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph.get(),
                         {conv_op, elementwise_add_op, elementwise_add_op_1,
                          elementwise_add_out});
  };
  gpd(graph.get(), handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add2_act_fuse_pass,
              paddle::framework::ir::ConvElementwiseAdd2ActFusePass);
