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

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

ConvConcatReLUFusePass::ConvConcatReLUFusePass() {
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
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("concat"))
      .AddInput("X")  // Input("X"): vector<tensors>
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumGE(0)
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

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
      PADDLE_ENFORCE_EQ(prev_op_node.size(), 1,
                        platform::errors::InvalidArgument(
                            "Node(%s) input size(%d) must be 1.", node->Name(),
                            prev_op_node.size()));
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
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
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

REGISTER_PASS_CAPABILITY(conv_concat_relu_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("concat", 0)
            .EQ("relu", 0));
