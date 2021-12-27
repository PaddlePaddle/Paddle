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
  return CreateBaseOp(graph, node, "popart_reducemean",
                      {GetInputVarNode("X", node)},
                      {GetOutputVarNode("Out", node)},
                      {
                          {"keepdims", int64_t{0}},
                      });
}

Node *pow_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  if (op->HasInput("FactorTensor") && !op->Input("FactorTensor").empty()) {
    return CreateBaseOp(
        graph, node, "popart_pow",
        {GetInputVarNode("X", node), GetInputVarNode("FactorTensor", node)},
        node->outputs);
  } else {
    // Op(pow) -> Op(Constant)->Var(const_out)->Op(Pow)
    auto value_ = BOOST_GET_CONST(float, op->GetAttr("factor"));
    auto attrs =
        MakeConstAttrMapFromValue<float>(value_, {1}, ONNXDataType::FLOAT);
    auto new_node_const = CreateConst(graph, node, {}, {}, attrs);
    return CreateBaseOp(graph, node, "popart_pow", {GetInputVarNode("X", node),
                                                    new_node_const->outputs[0]},
                        node->outputs);
  }
}

Node *mul_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto x_num_col_dims = BOOST_GET_CONST(int, op->GetAttr("x_num_col_dims"));
  auto y_num_col_dims = BOOST_GET_CONST(int, op->GetAttr("y_num_col_dims"));
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
  auto x_flatten =
      CreateBaseOp(graph, node, "popart_flatten", {GetInputVarNode("X", node)},
                   {}, {{"axis", int64_t(x_num_col_dims)}});
  auto y_flatten =
      CreateBaseOp(graph, node, "popart_flatten", {GetInputVarNode("Y", node)},
                   {}, {{"axis", int64_t(y_num_col_dims)}});
  auto matmul =
      CreateBaseOp(graph, node, "popart_matmul",
                   {x_flatten->outputs[0], y_flatten->outputs[0]}, {}, {});

  auto reshape_const = CreateConst(
      graph, node, {}, {},
      {{"value", reshape_shape_},
       {"dims", std::vector<int64_t>{int64_t(reshape_shape_.size())}},
       {"dtype", ONNXDataType::INT64}});
  return CreateBaseOp(graph, node, "popart_reshape",
                      {matmul->outputs[0], reshape_const->outputs[0]},
                      node->outputs, {});
}

Node *matmul_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto transpose_x = BOOST_GET_CONST(bool, op->GetAttr("transpose_X"));
  auto transpose_y = BOOST_GET_CONST(bool, op->GetAttr("transpose_Y"));
  auto alpha = BOOST_GET_CONST(float, op->GetAttr("alpha"));
  auto x_shape = GetInputVarNode("X", node)->Var()->GetShape();
  auto y_shape = GetInputVarNode("Y", node)->Var()->GetShape();

  int x_rank = x_shape.size();
  std::vector<int64_t> perm;
  if (x_rank == 1) {
    perm = std::vector<int64_t>{0};
  } else if (x_rank == 2) {
    return CreateGemm(graph, node,
                      {GetInputVarNode("X", node), GetInputVarNode("Y", node)},
                      node->outputs, transpose_x, transpose_y, alpha);
  } else if (x_rank == 3) {
    perm = std::vector<int64_t>{0, 2, 1};
  } else if (x_rank == 4) {
    perm = std::vector<int64_t>{0, 1, 3, 2};
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "op matmul with input rank == %d", x_rank));
  }

  Node *x_node = GetInputVarNode("X", node);
  Node *y_node = GetInputVarNode("Y", node);
  if (transpose_x) {
    x_node = CreateBaseOp(graph, node, "popart_transpose",
                          {GetInputVarNode("X", node)}, {}, {{"perm", perm}});
    x_node = x_node->outputs[0];
  }
  if (transpose_y) {
    y_node = CreateBaseOp(graph, node, "popart_transpose",
                          {GetInputVarNode("Y", node)}, {}, {{"perm", perm}});
    y_node = y_node->outputs[0];
  }
  if (is_float_equal(alpha, 1.0)) {
    auto o_node =
        CreateBaseOp(graph, node, "popart_matmul", {x_node, y_node}, {});
    auto attr = MakeConstAttrMapFromValue(alpha, {1}, ONNXDataType::FLOAT);
    auto const_node = CreateConst(graph, node, {}, {}, attr);
    return CreateBaseOp(graph, node, "popart_mul",
                        {o_node->outputs[0], const_node->outputs[0]},
                        node->outputs);
  } else {
    return CreateBaseOp(graph, node, "popart_matmul", {x_node, y_node},
                        node->outputs);
  }
}

Node *sum_handler(Graph *graph, Node *node) {
  return CreateBaseOp(graph, node, "popart_sum", node->inputs, node->outputs);
}

Node *softmax_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto axis = BOOST_GET_CONST(int, op->GetAttr("axis"));
  return CreateSoftmaxOpset11(graph, node, node->inputs, node->outputs, axis);
}

Node *scale_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto scale_ = BOOST_GET_CONST(float, op->GetAttr("scale"));
  auto bias_ = BOOST_GET_CONST(float, op->GetAttr("bias"));
  auto bias_after_scale_ =
      BOOST_GET_CONST(bool, op->GetAttr("bias_after_scale"));
  auto data_type_ = GetInputVarNode("X", node)->Var()->GetDataType();

  auto new_node_bias_var =
      CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{bias_}},
                                        {"dims", std::vector<int64_t>{1}},
                                        {"dtype", ONNXDataType::FLOAT}});
  new_node_bias_var = new_node_bias_var->outputs[0];

  Node *new_node_scale_var = nullptr;
  if (op->HasInput("ScaleTensor") && !op->Input("ScaleTensor").empty()) {
    new_node_scale_var = GetInputVarNode("ScaleTensor", node);
  } else {
    new_node_scale_var =
        CreateConst(graph, node, {}, {}, {{"value", std::vector<float>{scale_}},
                                          {"dims", std::vector<int64_t>{1}},
                                          {"dtype", ONNXDataType::FLOAT}});
    new_node_scale_var = new_node_scale_var->outputs[0];
  }

  // convert to float32
  auto new_node_cast =
      CreateCast(graph, node, {GetInputVarNode("X", node)}, {},
                 static_cast<int>(framework::proto::VarType::FP32));
  Node *result = nullptr;
  if (bias_after_scale_) {
    auto new_node_mul =
        CreateBaseOp(graph, node, "popart_mul",
                     {new_node_cast->outputs[0], new_node_scale_var}, {}, {});
    result =
        CreateBaseOp(graph, node, "popart_add",
                     {new_node_mul->outputs[0], new_node_bias_var}, {}, {});
  } else {
    auto new_node_add =
        CreateBaseOp(graph, node, "popart_add",
                     {new_node_cast->outputs[0], new_node_bias_var}, {}, {});
    result =
        CreateBaseOp(graph, node, "popart_mul",
                     {new_node_add->outputs[0], new_node_scale_var}, {}, {});
  }
  auto result_after_cast =
      CreateCast(graph, node, result->outputs, node->outputs,
                 static_cast<int>(data_type_));
  return result_after_cast;
}

Node *cross_entropy2_handler(Graph *graph, Node *node) {
  auto *op = node->Op();
  auto ignoreIndex = BOOST_GET_CONST(int, op->GetAttr("ignore_index"));
  auto new_cast = CreateCast(graph, node, {GetInputVarNode("Label", node)}, {},
                             framework::proto::VarType::INT32);
  auto label_shape_ = GetInputVarNode("Label", node)->Var()->GetShape();
  if (label_shape_.size() == 1) {
    return CreateBaseOp(graph, node, "popart_nllloss",
                        {GetInputVarNode("X", node), new_cast->outputs[0]},
                        {GetOutputVarNode("Y", node)},
                        {
                            {"ignoreIndex", ignoreIndex},
                        });
  } else {
    std::vector<int64_t> new_shape_{label_shape_[0]};
    auto const_before_loss = CreateBaseOp(
        graph, node, "popart_constant", {}, {},
        {{"value", new_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(new_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});

    auto reshape_before_loss = CreateBaseOp(
        graph, node, "popart_reshape",
        {new_cast->outputs[0], const_before_loss->outputs[0]}, {}, {});

    auto nllloss = CreateBaseOp(
        graph, node, "popart_nllloss",
        {GetInputVarNode("X", node), reshape_before_loss->outputs[0]}, {},
        {
            {"ignoreIndex", ignoreIndex},
        });

    auto const_after_loss = CreateBaseOp(
        graph, node, "popart_constant", {}, {},
        {{"value", label_shape_},
         {"dims",
          std::vector<int64_t>{static_cast<int64_t>(label_shape_.size())}},
         {"dtype", ONNXDataType::INT64}});

    auto reshape_after_loss =
        CreateBaseOp(graph, node, "popart_reshape",
                     {nllloss->outputs[0], const_after_loss->outputs[0]},
                     {GetOutputVarNode("Y", node)}, {});
    return reshape_after_loss;
  }
}

REGISTER_HANDLER(mean, mean_handler);
REGISTER_HANDLER(pow, pow_handler);
REGISTER_HANDLER(mul, mul_handler);
REGISTER_HANDLER(matmul, matmul_handler);
REGISTER_HANDLER(sum, sum_handler);
REGISTER_HANDLER(softmax, softmax_handler);
REGISTER_HANDLER(scale, scale_handler);
REGISTER_HANDLER(cross_entropy2, cross_entropy2_handler);

}  // namespace
}  // namespace ipu
}  // namespace platform
}  // namespace paddle
