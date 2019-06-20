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

#include "paddle/fluid/lite/core/mir/fusion/conv_elementwise_add_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvElementwiseAddActivationFuser::BuildPattern() {
  // create input nodes.
  auto* input =
      VarNode("input")->assert_is_op_input(conv_type_, "Input")->AsInput();
  auto* filter =
      VarNode("filter")->assert_is_op_input(conv_type_, "Filter")->AsInput();
  auto* bias =
      VarNode("bias")->assert_is_op_input("elementwise_add", "Y")->AsInput();

  // create op nodes
  auto* conv2d =
      OpNode("conv2d", conv_type_)->assert_is_op(conv_type_)->AsIntermediate();
  auto* add = OpNode("add", "elementwise_add")
                  ->assert_is_op("elementwise_add")
                  ->AsIntermediate();
  auto* act =
      OpNode("act", act_type_)->assert_is_op(act_type_)->AsIntermediate();

  // create intermediate nodes
  auto* conv2d_out = VarNode("conv2d_out")
                         ->assert_is_op_output(conv_type_, "Output")
                         ->assert_is_op_input("elementwise_add", "X")
                         ->AsIntermediate();
  auto* add_out = VarNode("add_out")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input(act_type_, "X")
                      ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> conv2d_inputs{filter, input};
  std::vector<PMNode*> add_inputs{conv2d_out, bias};
  conv2d_inputs >> *conv2d >> *conv2d_out;
  add_inputs >> *add >> *add_out;
  *add_out >> *act >> *out;
}

void ConvElementwiseAddActivationFuser::InsertNewNode(
    SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create(conv_type_);
  auto conv_old = matched.at("conv2d")->stmt()->op();
  auto* scope = conv_old->scope();
  auto& valid_places = conv_old->valid_places();
  conv_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ConvElementwiseAddActivationFuser::GenOpDesc(
    const key2nodes_t& matched) {
  auto* desc = matched.at("conv2d")->stmt()->op_info();

  cpp::OpDesc op_desc = *desc;
  op_desc.SetType(conv_type_);
  op_desc.SetInput("Input", {matched.at("input")->arg()->name});
  op_desc.SetInput("Filter", {matched.at("filter")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});
  // Other inputs. See operators/conv_op.h
  std::vector<std::string> input_arg_names = desc->InputArgumentNames();

  if (std::find(input_arg_names.begin(), input_arg_names.end(),
                "ResidualData") != input_arg_names.end()) {
    op_desc.SetInput("ResidualData", desc->Input("ResidualData"));
  }
  // Only consider strides, padding, groups, dilations, fuse_relu for now
  op_desc.SetAttr("strides", desc->GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings", desc->GetAttr<std::vector<int>>("paddings"));
  op_desc.SetAttr("groups", desc->GetAttr<int>("groups"));
  op_desc.SetAttr("dilations", desc->GetAttr<std::vector<int>>("dilations"));
  // TODO(sangoly): support other activation types
  op_desc.SetAttr("fuse_relu", true);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
