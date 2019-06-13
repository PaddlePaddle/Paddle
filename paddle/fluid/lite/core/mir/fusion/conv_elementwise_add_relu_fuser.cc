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

#include "paddle/fluid/lite/core/mir/fusion/conv_elementwise_add_relu_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ConvElementwiseAddReLUFuser::BuildPattern() {
  // create input nodes.
  auto* input = VarNode("input");
  auto* filter = VarNode("filter");
  auto* bias = VarNode("bias");

  // create op nodes
  auto* conv2d = OpNode("conv2d", "conv2d");
  auto* add = OpNode("add", "elementwise_add");
  auto* relu = OpNode("relu", "relu");

  // create intermediate nodes
  auto* conv2d_out = VarNode("conv2d_out");
  auto* add_out = VarNode("add_out");

  // create output node
  auto* out = VarNode("output");

  // create topology.
  std::vector<PMNode*> conv2d_inputs{filter, input};
  std::vector<PMNode*> add_inputs{conv2d_out, bias};
  conv2d_inputs >> *conv2d >> *conv2d_out;
  add_inputs >> *add >> *add_out;
  *add_out >> *relu >> *out;

  // Some op specialities.
  conv2d_out->AsIntermediate();
  add_out->AsIntermediate();
  conv2d->AsIntermediate();
  add->AsIntermediate();
  relu->AsIntermediate();
}

void ConvElementwiseAddReLUFuser::InsertNewNode(SSAGraph* graph,
                                                const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto conv_op = LiteOpRegistry::Global().Create("conv2d");
  auto conv_old = matched.at("conv2d")->stmt()->op;
  auto* scope = conv_old->scope();
  auto& valid_places = conv_old->valid_places();
  conv_op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(conv_op, valid_places);

  IR_NODE_LINK_TO(matched.at("input"), new_op_node);
  IR_NODE_LINK_TO(matched.at("filter"), new_op_node);
  IR_NODE_LINK_TO(matched.at("bias"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ConvElementwiseAddReLUFuser::GenOpDesc(const key2nodes_t& matched) {
  auto* desc = matched.at("conv2d")->stmt()->op_info();

  cpp::OpDesc op_desc;
  op_desc.SetType("conv2d");
  op_desc.SetInput("Input", {matched.at("input")->arg()->name});
  op_desc.SetInput("Filter", {matched.at("filter")->arg()->name});
  op_desc.SetInput("Bias", {matched.at("bias")->arg()->name});
  op_desc.SetOutput("Output", {matched.at("output")->arg()->name});
  // Other inputs. See operators/conv_op.h
  std::vector<std::string> input_arg_names = desc->InputArgumentNames();
  for (auto name : input_arg_names) LOG(INFO) << name;

  if (std::find(input_arg_names.begin(), input_arg_names.end(),
                "ResidualData") != input_arg_names.end()) {
    op_desc.SetInput("ResidualData", desc->Input("ResidualData"));
  }

  // Only consider strides, padding, groups, dilations, fuse_relu for now
  op_desc.SetAttr("strides", desc->GetAttr<std::vector<int>>("strides"));
  op_desc.SetAttr("paddings", desc->GetAttr<std::vector<int>>("paddings"));
  op_desc.SetAttr("groups", desc->GetAttr<int>("groups"));
  op_desc.SetAttr("dilations", desc->GetAttr<std::vector<int>>("dilations"));
  op_desc.SetAttr("fuse_relu", true);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
