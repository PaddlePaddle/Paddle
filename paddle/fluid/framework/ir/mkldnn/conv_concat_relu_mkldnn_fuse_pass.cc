// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/conv_concat_relu_mkldnn_fuse_pass.h"
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void ConvConcatReLUFusePass::FindConcatWithConvs(
    ir::Graph* graph,
    std::unordered_map<const Node*, int>* concat_with_convs_counter) const {
  GraphPatternDetector gpd;
  patterns::ConcatReLU concat_relu_pattern{gpd.mutable_pattern(),
                                           "concat_relu"};
  concat_relu_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Find Concats with Convs";
    GET_IR_NODE_FROM_SUBGRAPH(concat_op, concat_op, concat_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu_op, relu_op, concat_relu_pattern);

    auto concat_inputs = concat_op->inputs;

    for (auto node : concat_inputs) {
      auto prev_op_node = node->inputs;
      PADDLE_ENFORCE_EQ(prev_op_node.size(), 1);
      auto* conv_op = prev_op_node[0];
      if (conv_op->Op()->Type() != "conv2d") return;

      FuseOptions fuse_option = FindFuseOption(*conv_op, *relu_op);
      if (fuse_option == DO_NOT_FUSE) {
        return;
      }
    }

    (*concat_with_convs_counter)[concat_op] = concat_inputs.size();
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

void ConvConcatReLUFusePass::FuseConvConcatReLU(
    ir::Graph* graph,
    std::unordered_map<const Node*, int>* concat_with_convs_counter) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::ConvConcatReLU conv_concat_relu(pattern, name_scope_);
  conv_concat_relu();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvConcatReLU fuse";

    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_concat_relu);
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_concat_relu);
    GET_IR_NODE_FROM_SUBGRAPH(concat_op, concat_op, conv_concat_relu);
    GET_IR_NODE_FROM_SUBGRAPH(concat_out, concat_out, conv_concat_relu);
    GET_IR_NODE_FROM_SUBGRAPH(relu_op, relu_op, conv_concat_relu);
    GET_IR_NODE_FROM_SUBGRAPH(relu_out, relu_out, conv_concat_relu);

    if (!concat_with_convs_counter->count(concat_op)) {
      VLOG(4) << "this concat has input from non-conv2d operator";
      return;
    }

    // Transform Conv node into ConvReLU node.
    OpDesc* conv_desc = conv_op->Op();
    conv_desc->SetAttr("fuse_activation", std::string("relu"));

    // Remove ReLU when all Convs were transformed.
    auto number_of_unfused_convs_left =
        --(*concat_with_convs_counter)[concat_op];
    if (number_of_unfused_convs_left == 0) {
      OpDesc* concat_desc = concat_op->Op();
      concat_desc->SetOutput("Out",
                             std::vector<std::string>({relu_out->Name()}));
      GraphSafeRemoveNodes(graph, {relu_op, concat_out});
      IR_NODE_LINK_TO(concat_op, relu_out);
    }

    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

void ConvConcatReLUFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init(name_scope_, graph);

  std::unordered_map<const Node*, int> concat_with_convs_counter;
  FindConcatWithConvs(graph, &concat_with_convs_counter);
  FuseConvConcatReLU(graph, &concat_with_convs_counter);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_concat_relu_mkldnn_fuse_pass,
              paddle::framework::ir::ConvConcatReLUFusePass);
