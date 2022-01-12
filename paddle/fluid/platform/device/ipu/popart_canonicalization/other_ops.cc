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
  auto new_node = CreateBaseOp(graph, node, "popart_custom_op", node->inputs,
                               node->outputs, attrs);
  return new_node;
}

Node *print_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto print_phase = BOOST_GET_CONST(std::string, op->GetAttr("print_phase"));
  int64_t print_gradient = 0;
  if (print_phase != "forward") {
    print_gradient = 1;
  }
  auto title = BOOST_GET_CONST(std::string, op->GetAttr("message"));
  if (title.empty()) {
    title = GetInputVarNode("In", node)->Var()->Name();
  }
  auto attrs =
      AttributeMap{{"print_gradient", print_gradient}, {"title", title}};
  return CreateBaseOp(graph, node, "popart_printtensor", node->inputs,
                      node->outputs, attrs);
}

Node *popart_optimizer_handler(Graph *graph, Node *node) { return nullptr; }

Node *checkpointoutput_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_checkpointoutput", node->inputs,
                      node->outputs);
}

REGISTER_HANDLER(custom_op, custom_op_handler);
REGISTER_HANDLER(print, print_handler);
REGISTER_HANDLER(popart_optimizer, popart_optimizer_handler);
REGISTER_HANDLER(checkpointoutput, checkpointoutput_handler);

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle
