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

Node *activation_op_handler(Graph *graph, Node *node, const std::string &type) {
  auto new_node = CreateBaseOp(graph, node, type, {GetInputVarNode("X", node)},
                               node->outputs);
  return new_node;
}

Node *relu_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_relu");
}

Node *tanh_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_tanh");
}

Node *log_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_log");
}

Node *sigmoid_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sigmoid");
}

Node *sqrt_handler(Graph *graph, Node *node) {
  return activation_op_handler(graph, node, "popart_sqrt");
}

Node *gelu_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto approximate_ = BOOST_GET_CONST(bool, op->GetAttr("approximate"));
  if (approximate_) {
    return activation_op_handler(graph, node, "popart_gelu_v2");
  } else {
    auto sqrt2 = CreateConst(graph, node, {}, {},
                             {{"value", std::vector<float>{1.4142135623730951}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", GetOutputVarDtype(node)}});
    auto zero_point_five =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{0.5}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", GetOutputVarDtype(node)}});
    auto one =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{1}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", GetOutputVarDtype(node)}});
    auto div =
        CreateBaseOp(graph, node, "popart_div",
                     {GetInputVarNode("X", node), sqrt2->outputs[0]}, {}, {});
    auto erf =
        CreateBaseOp(graph, node, "popart_erf", {div->outputs[0]}, {}, {});
    auto add = CreateBaseOp(graph, node, "popart_add",
                            {erf->outputs[0], one->outputs[0]}, {}, {});
    auto mul1 =
        CreateBaseOp(graph, node, "popart_mul",
                     {GetInputVarNode("X", node), add->outputs[0]}, {}, {});
    return CreateBaseOp(graph, node, "popart_mul",
                        {mul1->outputs[0], zero_point_five->outputs[0]},
                        {GetOutputVarNode("Out", node)}, {});
  }
}

Node *log_softmax_handler(Graph *graph, Node *node) {
  auto axis = BOOST_GET_CONST(int, node->Op()->GetAttr("axis"));
  auto new_softmax = CreateSoftmaxOpset11(graph, node, node->inputs, {}, axis);
  return CreateBaseOp(graph, node, "popart_log", new_softmax->outputs,
                      node->outputs);
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(relu, relu_handler);
REGISTER_HANDLER(tanh, tanh_handler);
REGISTER_HANDLER(log, log_handler);
REGISTER_HANDLER(sigmoid, sigmoid_handler);
REGISTER_HANDLER(sqrt, sqrt_handler);
REGISTER_HANDLER(gelu, gelu_handler);
REGISTER_HANDLER(log_softmax, log_softmax_handler);
