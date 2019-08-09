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

#include "paddle/fluid/lite/core/mir/fusion/elementwise_add_activation_fuser.h"
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void ElementwiseAddActivationFuser::BuildPattern() {
  // create input nodes.
  auto* x = VarNode("x")->assert_is_op_input("elementwise_add", "X")->AsInput();
  auto* y = VarNode("y")->assert_is_op_input("elementwise_add", "Y")->AsInput();

  // create op nodes
  auto* add = OpNode("add", "elementwise_add")
                  ->assert_is_op("elementwise_add")
                  ->AsIntermediate();
  auto* act =
      OpNode("act", act_type_)->assert_is_op(act_type_)->AsIntermediate();

  // create intermediate nodes
  auto* add_out = VarNode("add_out")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input(act_type_, "X")
                      ->AsIntermediate();

  // create output node
  auto* out =
      VarNode("output")->assert_is_op_output(act_type_, "Out")->AsOutput();

  // create topology.
  std::vector<PMNode*> add_inputs{x, y};
  add_inputs >> *add >> *add_out;
  *add_out >> *act >> *out;
}

void ElementwiseAddActivationFuser::InsertNewNode(SSAGraph* graph,
                                                  const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto op =
      LiteOpRegistry::Global().Create("fusion_elementwise_add_activation");
  auto old_op = matched.at("add")->stmt()->op();
  auto* scope = old_op->scope();
  auto& valid_places = old_op->valid_places();
  op->Attach(op_desc, scope);

  auto* new_op_node = graph->GraphCreateInstructNode(op, valid_places);

  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(matched.at("y"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("output"));
}

cpp::OpDesc ElementwiseAddActivationFuser::GenOpDesc(
    const key2nodes_t& matched) {
  auto* desc = matched.at("add")->stmt()->op_info();

  cpp::OpDesc op_desc;
  op_desc.SetType("fusion_elementwise_add_activation");
  op_desc.SetInput("X", {matched.at("x")->arg()->name});
  op_desc.SetInput("Y", {matched.at("y")->arg()->name});
  op_desc.SetOutput("Out", {matched.at("output")->arg()->name});

  op_desc.SetAttr("axis", desc->GetAttr<int>("axis"));
  op_desc.SetAttr("act_type", act_type_);
  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
