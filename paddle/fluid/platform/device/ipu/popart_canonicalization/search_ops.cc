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

Node *topk_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto attrs = AttributeMap{};

  int axis_ = -1;
  if (op->HasAttr("axis")) {
    axis_ = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  }
  if (axis_ == -1) {
    auto shape = GetInputVarNode("X", node)->Var()->GetShape();
    int rank = shape.size();
    if (rank < 1) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "The dimension of the shape of topK input should be large than 1"));
    }
    axis_ = rank - 1;
  }
  int64_t axis = int64_t{axis_};
  attrs.emplace("axis", axis);

  bool largest = true;
  if (op->HasAttr("largest")) {
    largest = PADDLE_GET_CONST(bool, op->GetAttr("largest"));
  }
  if (largest) {
    // defaults to 1, largest values
    attrs.emplace("largest", 1);
  } else {
    attrs.emplace("largest", 0);
  }

  bool sorted = true;
  if (op->HasAttr("sorted")) {
    sorted = PADDLE_GET_CONST(bool, op->GetAttr("sorted"));
  }
  if (sorted) {
    // defaults to 1, sorted results
    attrs.emplace("sorted", 1);
  } else {
    attrs.emplace("sorted", 0);
  }

  Node *var_x = GetInputVarNode("X", node);
  Node *var_k = nullptr;
  if (!op->Input("K").empty()) {
    var_k = GetInputVarNode("K", node);
  } else {
    auto k = PADDLE_GET_CONST(int, op->GetAttr("k"));
    auto *op_k = CreateConst(graph,
                             node,
                             {},
                             {},
                             {{"value", std::vector<int64_t>{k}},
                              {"dims", std::vector<int64_t>{1}},
                              {"dtype", ONNXDataType::INT64}});
    var_k = op_k->outputs[0];
  }

  auto *var_i = MakeVarNode(graph, node);
  CreateBaseOp(graph,
               node,
               "popart_topk",
               {var_x, var_k},
               {GetOutputVarNode("Out", node), var_i},
               {{"axis", int64_t{axis}},
                {"largest", int64_t{largest}},
                {"sorted", int64_t{sorted}}});
  return CreateCast(graph,
                    node,
                    {var_i},
                    {GetOutputVarNode("Indices", node)},
                    VarType::INT32);
}

Node *argsort_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto x_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto axis_ = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto descending_ = PADDLE_GET_CONST(bool, op->GetAttr("descending"));
  if (axis_ < 0) {
    axis_ = axis_ + x_shape.size();
  }
  auto *dim_size = CreateConst(graph,
                               node,
                               std::vector<int64_t>{x_shape[axis_]},
                               {1},
                               ONNXDataType::INT64)
                       ->outputs.front();
  int64_t largest = descending_ ? 1 : 0;
  return CreateBaseOp(
      graph,
      node,
      "popart_topk",
      {GetInputVarNode("X", node), dim_size},
      {GetOutputVarNode("Out", node), GetOutputVarNode("Indices", node)},
      {
          {"axis", int64_t{axis_}},
          {"largest", int64_t{largest}},
          {"sorted", int64_t{0}},
      });
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(top_k, topk_handler);
REGISTER_HANDLER(top_k_v2, topk_handler);
REGISTER_HANDLER(argsort, argsort_handler);
