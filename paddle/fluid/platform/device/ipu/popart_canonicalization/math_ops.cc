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

Node *mean_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph,
                      node,
                      "popart_reducemean",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {
                          {"keepdims", int64_t{0}},
                      });
}

Node *pow_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  if (!op->Input("FactorTensor").empty()) {
    return CreateBaseOp(
        graph,
        node,
        "popart_pow",
        {GetInputVarNode("X", node), GetInputVarNode("FactorTensor", node)},
        node->outputs);
  } else {
    // Op(pow) -> Op(Constant)->Var(const_out)->Op(Pow)
    auto value_ = PADDLE_GET_CONST(float, op->GetAttr("factor"));
    auto new_node_const = CreateConst(graph,
                                      node,
                                      std::vector<decltype(value_)>{value_},
                                      {1},
                                      GetOutputVarDType(node));

    return CreateBaseOp(
        graph,
        node,
        "popart_pow",
        {GetInputVarNode("X", node), new_node_const->outputs[0]},
        node->outputs);
  }
}

Node *mul_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto x_num_col_dims = PADDLE_GET_CONST(int, op->GetAttr("x_num_col_dims"));
  auto y_num_col_dims = PADDLE_GET_CONST(int, op->GetAttr("y_num_col_dims"));
  auto x_shape_ = GetInputVarNode("X", node)->Var()->GetShape();
  auto y_shape_ = GetInputVarNode("Y", node)->Var()->GetShape();

  // build the shape for reshape
  std::vector<int64_t> reshape_shape_{};
  for (int left = 0; left < x_num_col_dims; left++) {
    reshape_shape_.push_back(int64_t(x_shape_[left]));
  }
  for (int right = y_num_col_dims; right < y_shape_.size(); right++) {
    reshape_shape_.push_back(int64_t(y_shape_[right]));
  }
  auto x_flatten = CreateBaseOp(graph,
                                node,
                                "popart_flatten",
                                {GetInputVarNode("X", node)},
                                {},
                                {{"axis", int64_t(x_num_col_dims)}});
  auto y_flatten = CreateBaseOp(graph,
                                node,
                                "popart_flatten",
                                {GetInputVarNode("Y", node)},
                                {},
                                {{"axis", int64_t(y_num_col_dims)}});
  auto matmul = CreateBaseOp(graph,
                             node,
                             "popart_matmul",
                             {x_flatten->outputs[0], y_flatten->outputs[0]},
                             {},
                             {});

  auto reshape_const = CreateConst(
      graph,
      node,
      {},
      {},
      {{"value", reshape_shape_},
       {"dims", std::vector<int64_t>{int64_t(reshape_shape_.size())}},
       {"dtype", ONNXDataType::INT64}});
  return CreateBaseOp(graph,
                      node,
                      "popart_reshape",
                      {matmul->outputs[0], reshape_const->outputs[0]},
                      node->outputs,
                      {});
}

Node *matmul_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto transpose_x = PADDLE_GET_CONST(bool, op->GetAttr("transpose_X"));
  auto transpose_y = PADDLE_GET_CONST(bool, op->GetAttr("transpose_Y"));
  auto alpha = PADDLE_GET_CONST(float, op->GetAttr("alpha"));
  Node *x_node = GetInputVarNode("X", node);
  Node *y_node = GetInputVarNode("Y", node);
  int x_rank = x_node->Var()->GetShape().size();
  int y_rank = y_node->Var()->GetShape().size();

  auto gen_perm = [](const int rank) -> std::vector<int64_t> {
    std::vector<int64_t> perm;
    if (rank == 1) {
      perm = std::vector<int64_t>{0};
    } else if (rank == 2) {
      perm = std::vector<int64_t>{1, 0};
    } else if (rank == 3) {
      perm = std::vector<int64_t>{0, 2, 1};
    } else if (rank == 4) {
      perm = std::vector<int64_t>{0, 1, 3, 2};
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "op matmul with input rank == %d", rank));
    }
    return perm;
  };

  if (x_rank == 2) {
    if (!transpose_x && !transpose_y && is_float_equal(alpha, 1.0f)) {
      return CreateBaseOp(
          graph,
          node,
          "popart_matmul",
          {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
          node->outputs);
    }
    return CreateGemm(graph,
                      node,
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      node->outputs,
                      transpose_x,
                      transpose_y,
                      alpha);
  }

  if (transpose_x) {
    auto perm = gen_perm(x_rank);
    x_node = CreateBaseOp(graph,
                          node,
                          "popart_transpose",
                          {GetInputVarNode("X", node)},
                          {},
                          {{"perm", perm}});
    x_node = x_node->outputs[0];
  }
  if (transpose_y) {
    auto perm = gen_perm(y_rank);
    y_node = CreateBaseOp(graph,
                          node,
                          "popart_transpose",
                          {GetInputVarNode("Y", node)},
                          {},
                          {{"perm", perm}});
    y_node = y_node->outputs[0];
  }
  if (is_float_equal(alpha, 1.0)) {
    return CreateBaseOp(
        graph, node, "popart_matmul", {x_node, y_node}, node->outputs);
  } else {
    auto o_node =
        CreateBaseOp(graph, node, "popart_matmul", {x_node, y_node}, {});
    auto const_node = CreateConst(graph,
                                  node,
                                  std::vector<decltype(alpha)>{alpha},
                                  {1},
                                  GetOutputVarDType(node));
    return CreateBaseOp(graph,
                        node,
                        "popart_mul",
                        {o_node->outputs[0], const_node->outputs[0]},
                        node->outputs);
  }
}

Node *sum_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_sum", node->inputs, node->outputs);
}

Node *softmax_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  int axis = -1;
  if (op->HasAttr("axis")) {
    axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  }
  return CreateSoftmaxOpset11(graph, node, node->inputs, node->outputs, axis);
}

Node *scale_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto bias_ = PADDLE_GET_CONST(float, op->GetAttr("bias"));
  auto bias_after_scale_ =
      PADDLE_GET_CONST(bool, op->GetAttr("bias_after_scale"));
  auto data_type_ = GetInputVarNode("X", node)->Var()->GetDataType();

  auto cast =
      CreateCast(graph, node, {GetInputVarNode("X", node)}, {}, VarType::FP32);

  Node *result = nullptr;
  if (op->InputArgumentNames().size() > 1) {
    auto scale = GetInputVarNode("ScaleTensor", node);
    if (is_float_equal(bias_, 0.0)) {
      result = CreateBaseOp(
          graph, node, "popart_mul", {cast->outputs[0], scale}, {}, {});
    } else {
      auto bias = CreateConst(graph,
                              node,
                              {},
                              {},
                              {{"value", std::vector<float>{bias_}},
                               {"dims", std::vector<int64_t>{1}},
                               {"dtype", ONNXDataType::FLOAT}});
      bias = bias->outputs[0];
      if (bias_after_scale_) {
        auto mul = CreateBaseOp(
            graph, node, "popart_mul", {cast->outputs[0], scale}, {}, {});
        result = CreateBaseOp(
            graph, node, "popart_add", {mul->outputs[0], bias}, {}, {});
      } else {
        auto add = CreateBaseOp(
            graph, node, "popart_add", {cast->outputs[0], bias}, {}, {});
        result = CreateBaseOp(
            graph, node, "popart_mul", {add->outputs[0], scale}, {}, {});
      }
    }
  } else {
    auto scale_ = PADDLE_GET_CONST(float, op->GetAttr("scale"));
    if (is_float_equal(bias_, 0.0) && is_float_equal(scale_, 1.0)) {
      return CreateBaseOp(graph,
                          node,
                          "popart_identity",
                          {GetInputVarNode("X", node)},
                          node->outputs,
                          {});
    } else if (is_float_equal(scale_, 1.0)) {
      auto bias = CreateConst(graph,
                              node,
                              {},
                              {},
                              {{"value", std::vector<float>{bias_}},
                               {"dims", std::vector<int64_t>{1}},
                               {"dtype", ONNXDataType::FLOAT}});
      result = CreateBaseOp(graph,
                            node,
                            "popart_add",
                            {cast->outputs[0], bias->outputs[0]},
                            {},
                            {});
    } else if (is_float_equal(bias_, 0.0)) {
      auto scale = CreateConst(graph,
                               node,
                               {},
                               {},
                               {{"value", std::vector<float>{scale_}},
                                {"dims", std::vector<int64_t>{1}},
                                {"dtype", ONNXDataType::FLOAT}});
      result = CreateBaseOp(graph,
                            node,
                            "popart_mul",
                            {cast->outputs[0], scale->outputs[0]},
                            {},
                            {});
    } else {
      auto bias = CreateConst(graph,
                              node,
                              {},
                              {},
                              {{"value", std::vector<float>{bias_}},
                               {"dims", std::vector<int64_t>{1}},
                               {"dtype", ONNXDataType::FLOAT}});
      auto scale = CreateConst(graph,
                               node,
                               {},
                               {},
                               {{"value", std::vector<float>{scale_}},
                                {"dims", std::vector<int64_t>{1}},
                                {"dtype", ONNXDataType::FLOAT}});
      if (bias_after_scale_) {
        auto mul = CreateBaseOp(graph,
                                node,
                                "popart_mul",
                                {cast->outputs[0], scale->outputs[0]},
                                {},
                                {});
        result = CreateBaseOp(graph,
                              node,
                              "popart_add",
                              {mul->outputs[0], bias->outputs[0]},
                              {},
                              {});
      } else {
        auto add = CreateBaseOp(graph,
                                node,
                                "popart_add",
                                {cast->outputs[0], bias->outputs[0]},
                                {},
                                {});
        result = CreateBaseOp(graph,
                              node,
                              "popart_mul",
                              {add->outputs[0], scale->outputs[0]},
                              {},
                              {});
      }
    }
  }
  auto result_after_cast =
      CreateCast(graph, node, result->outputs, node->outputs, data_type_);
  return result_after_cast;
}

Node *cumsum_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto exclusive = PADDLE_GET_CONST(bool, op->GetAttr("exclusive"));
  int64_t popart_exclusive = 1 ? exclusive : 0;
  auto reverse = PADDLE_GET_CONST(bool, op->GetAttr("reverse"));
  int64_t popart_reverse = 1 ? reverse : 0;
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto axis_node = CreateConst(graph,
                               node,
                               {},
                               {},
                               {{"value", std::vector<int64_t>{axis}},
                                {"dims", std::vector<int64_t>{1}},
                                {"dtype", ONNXDataType::INT64}});
  Node *input_x = nullptr;
  auto data_type_ = GetInputVarNode("X", node)->Var()->GetDataType();
  bool need_cast = data_type_ != VarType::FP32;
  std::vector<Node *> cumsum_out;
  if (need_cast) {
    auto cast_x = CreateCast(
        graph, node, {GetInputVarNode("X", node)}, {}, VarType::FP32);
    input_x = cast_x->outputs[0];
  } else {
    input_x = GetInputVarNode("X", node);
    cumsum_out.emplace_back(GetOutputVarNode("Out", node));
  }
  auto cumsum_node = CreateBaseOp(
      graph,
      node,
      "popart_cumsum",
      {input_x, axis_node->outputs[0]},
      cumsum_out,
      {{"exclusive", popart_exclusive}, {"reverse", popart_reverse}});
  if (need_cast) {
    cumsum_node = CreateCast(graph,
                             node,
                             cumsum_node->outputs,
                             {GetOutputVarNode("Out", node)},
                             data_type_);
  }
  return cumsum_node;
}

Node *matmul_v2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto transpose_x = PADDLE_GET_CONST(bool, op->GetAttr("trans_x"));
  auto transpose_y = PADDLE_GET_CONST(bool, op->GetAttr("trans_y"));
  Node *x_node = GetInputVarNode("X", node);
  Node *y_node = GetInputVarNode("Y", node);
  int x_rank = x_node->Var()->GetShape().size();
  int y_rank = y_node->Var()->GetShape().size();

  auto gen_perm = [](const int rank) -> std::vector<int64_t> {
    std::vector<int64_t> perm;
    if (rank == 1) {
      perm = std::vector<int64_t>{0};
    } else if (rank == 2) {
      perm = std::vector<int64_t>{1, 0};
    } else if (rank == 3) {
      perm = std::vector<int64_t>{0, 2, 1};
    } else if (rank == 4) {
      perm = std::vector<int64_t>{0, 1, 3, 2};
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "op matmul with input rank == %d", rank));
    }
    return perm;
  };

  if (transpose_x) {
    auto perm = gen_perm(x_rank);
    x_node = CreateBaseOp(graph,
                          node,
                          "popart_transpose",
                          {GetInputVarNode("X", node)},
                          {},
                          {{"perm", perm}});
    x_node = x_node->outputs[0];
  }
  if (transpose_y) {
    auto perm = gen_perm(y_rank);
    y_node = CreateBaseOp(graph,
                          node,
                          "popart_transpose",
                          {GetInputVarNode("Y", node)},
                          {},
                          {{"perm", perm}});
    y_node = y_node->outputs[0];
  }

  return CreateBaseOp(
      graph, node, "popart_matmul", {x_node, y_node}, node->outputs);
}

Node *bmm_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph,
                      node,
                      "popart_matmul",
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      node->outputs);
}

Node *arg_max_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int64_t, op->GetAttr("axis"));
  return CreateBaseOp(graph,
                      node,
                      "popart_argmax",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {{"axis", axis}, {"keepdims", int64_t{0}}});
}

Node *arg_min_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int64_t, op->GetAttr("axis"));
  return CreateBaseOp(graph,
                      node,
                      "popart_argmin",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {{"axis", axis}, {"keepdims", int64_t{0}}});
}

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle

REGISTER_HANDLER(mean, mean_handler);
REGISTER_HANDLER(pow, pow_handler);
REGISTER_HANDLER(mul, mul_handler);
REGISTER_HANDLER(matmul, matmul_handler);
REGISTER_HANDLER(sum, sum_handler);
REGISTER_HANDLER(softmax, softmax_handler);
REGISTER_HANDLER(scale, scale_handler);
REGISTER_HANDLER(cumsum, cumsum_handler);
REGISTER_HANDLER(matmul_v2, matmul_v2_handler);
REGISTER_HANDLER(bmm, bmm_handler);
REGISTER_HANDLER(arg_max, arg_max_handler);
REGISTER_HANDLER(arg_min, arg_min_handler);
