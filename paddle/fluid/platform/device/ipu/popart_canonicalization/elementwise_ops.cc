// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

Node *elementwise_op_handler(Graph *graph, Node *node,
                             const std::string &type) {
  auto *op = node->Op();
  auto x_shape = GetInputVarNode("X", node)->Var()->GetShape();
  int64_t x_rank = x_shape.size();
  auto y_shape = GetInputVarNode("Y", node)->Var()->GetShape();
  int64_t y_rank = y_shape.size();

  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  if (axis == -1 || axis == x_rank - 1 || x_rank == y_rank) {
    auto new_node =
        CreateBaseOp(graph, node, type,
                     {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                     node->outputs);
    return new_node;
  } else {
    auto y_new_shape = std::vector<int64_t>(x_rank, 1);
    for (int i = axis; i < axis + y_rank; ++i) {
      y_new_shape[i] = y_shape[i - axis];
    }
    auto attrs = AttributeMap{
        {"value", y_new_shape},
        {"dims", std::vector<int64_t>{x_rank}},
        {"dtype", ONNXDataType::INT64},
    };
    // constant
    auto new_node_const = CreateConst(graph, node, {}, {}, attrs);
    // reshape
    auto new_node_reshape = CreateBaseOp(
        graph, node, "popart_reshape",
        {GetInputVarNode("Y", node), new_node_const->outputs[0]}, {});
    // elementwise_op
    auto new_node =
        CreateBaseOp(graph, node, type,
                     {GetInputVarNode("X", node), new_node_reshape->outputs[0]},
                     node->outputs);
    return new_node;
  }
}

Node *elementwise_add_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_add");
}

Node *elementwise_sub_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_sub");
}

Node *elementwise_div_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_div");
}

Node *elementwise_mul_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_mul");
}

Node *elementwise_min_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_min");
}

Node *elementwise_max_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_max");
}

Node *elementwise_pow_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_pow");
}

Node *elementwise_mod_handler(Graph *graph, Node *node) {
  return elementwise_op_handler(graph, node, "popart_mod");
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(elementwise_add, elementwise_add_handler);
REGISTER_HANDLER(elementwise_sub, elementwise_sub_handler);
REGISTER_HANDLER(elementwise_div, elementwise_div_handler);
REGISTER_HANDLER(elementwise_mul, elementwise_mul_handler);
REGISTER_HANDLER(elementwise_min, elementwise_min_handler);
REGISTER_HANDLER(elementwise_max, elementwise_max_handler);
REGISTER_HANDLER(elementwise_pow, elementwise_pow_handler);
REGISTER_HANDLER(elementwise_mod, elementwise_mod_handler);
