// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/device/ipu/popart_canonicalization/canonicalization_utils.h"
#include "paddle/fluid/platform/device/ipu/popart_canonicalization/op_builder.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace ipu {
namespace {

Node *custom_op_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto attrs = op->GetAttrMap();
  attrs.insert({"__op_type", node->Op()->Type()});
  auto new_node = CreateBaseOp(
      graph, node, "popart_custom_op", node->inputs, node->outputs, attrs);
  return new_node;
}

Node *print_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto print_output = node->outputs.front();
  auto print_input = node->inputs.front();
  if (print_output->outputs.size() == 0) {
    LOG(WARNING) << "The output of Print OP is not used on IPU graph. Setting "
                    "the input of Print as Output.";
    for (auto &subnode : print_input->outputs) {
      if (subnode == node) continue;
      ConnectNodes(print_output, subnode);
      DisConnectNodes(print_input, subnode);

      // replace node_name in op_desc
      std::vector<std::string> new_inputs;
      auto subnode_inmap = subnode->Op()->Inputs();
      for (auto &in_map : subnode_inmap) {
        if (std::find(in_map.second.begin(),
                      in_map.second.end(),
                      print_input->Name()) != in_map.second.end()) {
          std::transform(in_map.second.cbegin(),
                         in_map.second.cend(),
                         std::back_inserter(new_inputs),
                         [&](const std::string &node_name) {
                           if (node_name == print_input->Name()) {
                             return print_output->Name();
                           } else {
                             return node_name;
                           }
                         });
          subnode->Op()->SetInput(in_map.first, new_inputs);
          subnode->Op()->Flush();
        }
      }
    }
  }
  auto print_phase = PADDLE_GET_CONST(std::string, op->GetAttr("print_phase"));
  int64_t print_gradient = 0;
  if (print_phase != "forward") {
    print_gradient = 1;
  }
  auto title = PADDLE_GET_CONST(std::string, op->GetAttr("message"));
  if (title.empty()) {
    title = GetInputVarNode("In", node)->Var()->Name();
  }
  auto attrs =
      AttributeMap{{"print_gradient", print_gradient}, {"title", title}};
  return CreateBaseOp(
      graph, node, "popart_printtensor", node->inputs, node->outputs, attrs);
}

Node *popart_optimizer_handler(Graph *graph, Node *node) { return nullptr; }

Node *checkpointoutput_handler(Graph *graph, Node *node) {
  return CreateBaseOp(
      graph, node, "popart_checkpointoutput", node->inputs, node->outputs);
}

Node *custom_nll_loss_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto reduction = PADDLE_GET_CONST(int, op->GetAttr("reduction"));
  auto ignoreIndex = PADDLE_GET_CONST(std::string, op->GetAttr("ignoreIndex"));
  auto inputIsLogProbability =
      PADDLE_GET_CONST(bool, op->GetAttr("inputIsLogProbability"));
  if (ignoreIndex == "None") {
    return CreateBaseOp(graph,
                        node,
                        "popart_nllloss_v2",
                        node->inputs,
                        node->outputs,
                        {{"reduction", reduction},
                         {"inputIsLogProbability", inputIsLogProbability}});
  } else {
    return CreateBaseOp(graph,
                        node,
                        "popart_nllloss_v2",
                        node->inputs,
                        node->outputs,
                        {{"reduction", reduction},
                         {"ignoreIndex", std::atoi(ignoreIndex.c_str())},
                         {"inputIsLogProbability", inputIsLogProbability}});
  }
}

Node *identity_handler(Graph *graph, Node *node) {
  return CreateBaseOp(
      graph, node, "popart_identity", node->inputs, node->outputs);
}

Node *detach_handler(Graph *graph, Node *node) {
  return CreateBaseOp(
      graph, node, "popart_detach_v2", node->inputs, node->outputs);
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(custom_op, custom_op_handler);
REGISTER_HANDLER(print, print_handler);
REGISTER_HANDLER(popart_optimizer, popart_optimizer_handler);
REGISTER_HANDLER(checkpointoutput, checkpointoutput_handler);
REGISTER_HANDLER(custom_nll_loss, custom_nll_loss_handler);
REGISTER_HANDLER(identity, identity_handler);
REGISTER_HANDLER(detach, detach_handler);
