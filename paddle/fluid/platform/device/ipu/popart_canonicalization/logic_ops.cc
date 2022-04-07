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

Node *equal_handler(Graph *graph, Node *node) {
  auto new_node = CreateBaseOp(
      graph, node, "popart_equal",
      {GetInputVarNode("X", node), GetInputVarNode("Y", node)}, node->outputs);
  return new_node;
}

Node *logical_not_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_logical_not",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

Node *logical_or_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_logical_or",
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

Node *logical_and_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_logical_and",
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

Node *greater_than_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_greater",
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

Node *less_than_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_less",
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      {GetOutputVarNode("Out", node)}, {});
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(equal, equal_handler);
REGISTER_HANDLER(logical_not, logical_not_handler);
REGISTER_HANDLER(logical_or, logical_or_handler);
REGISTER_HANDLER(logical_and, logical_and_handler);
REGISTER_HANDLER(greater_than, greater_than_handler);
REGISTER_HANDLER(less_than, less_than_handler);
