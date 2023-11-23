/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kAdd = "elementwise_add";
const char *const kAddGrad = "elementwise_add_grad";
const char *const kSub = "elementwise_sub";
const char *const kSubGrad = "elementwise_sub_grad";
const char *const kMul = "elementwise_mul";
const char *const kMulGrad = "elementwise_mul_grad";
const char *const kDiv = "elementwise_div";
const char *const kDivGrad = "elementwise_div_grad";
const char *const kGreater = "greater_than";
const char *const kLessEqual = "less_equal";
const char *const kLessThan = "less_than";
const char *const kElementWisePow = "elementwise_pow";
const char *const kElementWisePowGrad = "elementwise_pow_grad";
const char *const kGreaterEqual = "greater_equal";

static void print_shape(const std::vector<int64_t> &shape,
                        const std::string &name) {
  std::cout << "---- shape of " << name << ": ";
  for (auto dim : shape) {
    std::cout << dim << ", ";
  }
  std::cout << std::endl;
}

static void print_shape(const std::vector<std::vector<int64_t>> &shape,
                        const std::string &name) {
  std::cout << "---- shape of " << name << ": ";
  for (auto dims : shape) {
    for (auto dim : dims) {
      std::cout << dim << ", ";
    }
    std::cout << "; ";
  }
  std::cout << std::endl;
}

#define BINARY_OP(name)                                                        \
  IMPLEMT_EQUIVALENCE_TRANS_FUNC(                                              \
      gcu_builder, node, map_inputs, running_mode, name##EquivalenceTrans) {   \
    std::vector<GcuOpPtr> inputs;                                              \
    if (map_inputs.count("X") != 0) {                                          \
      inputs.push_back(map_inputs["X"].at(0));                                 \
    } else {                                                                   \
      PADDLE_ENFORCE_EQ(                                                       \
          true, false, platform::errors::NotFound("lack of [X] gcu op"));      \
    }                                                                          \
    if (map_inputs.count("Y") != 0) {                                          \
      inputs.push_back(map_inputs["Y"].at(0));                                 \
    } else {                                                                   \
      PADDLE_ENFORCE_EQ(                                                       \
          true, false, platform::errors::NotFound("lack of [Y] gcu op"));      \
    }                                                                          \
    auto *op = node->Op();                                                     \
    auto lhs_shape = inputs[0]->GetType().GetShape();                          \
    auto rhs_shape = inputs[1]->GetType().GetShape();                          \
    print_shape(lhs_shape, "lhs");                                             \
    print_shape(rhs_shape, "rhs");                                             \
    if (lhs_shape == rhs_shape) {                                              \
      if (lhs_shape.size() != 4 ||                                             \
          (lhs_shape.size() == 4 && running_mode != RunningMode::ADAPTIVE)) {  \
        return std::make_shared<GcuOp>(builder::name(*inputs[0], *inputs[1])); \
      }                                                                        \
      auto lv = builder::Transpose(*inputs[0], {0, 2, 3, 1});                  \
      auto rv = builder::Transpose(*inputs[1], {0, 2, 3, 1});                  \
      auto res = builder::name(lv, rv);                                        \
      return std::make_shared<GcuOp>(                                          \
          builder::Transpose(builder::name(lv, rv), {0, 3, 1, 2}));            \
    }                                                                          \
    auto axis =                                                                \
        static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));      \
    auto lhs_rank = inputs[0]->GetType().GetRank();                            \
    auto rhs_rank = inputs[1]->GetType().GetRank();                            \
    std::map<std::string, GcuOp> op_map{{"X", *inputs[0]}, {"Y", *inputs[1]}}; \
    auto low = lhs_rank < rhs_rank ? "X" : "Y";                                \
    std::vector<int64_t> new_shape;                                            \
    int64_t iter = 0;                                                          \
    if (lhs_rank < rhs_rank) {                                                 \
      new_shape.assign(rhs_rank, 1);                                           \
      axis = axis > 0 ? axis : rhs_rank - lhs_rank;                            \
      for (int64_t i = axis; i < axis + lhs_rank; ++i) {                       \
        new_shape[i] = lhs_shape[iter++];                                      \
      }                                                                        \
    } else {                                                                   \
      new_shape.assign(lhs_rank, 1);                                           \
      axis = axis > 0 ? axis : lhs_rank - rhs_rank;                            \
      for (int64_t i = axis; i < axis + rhs_rank; ++i) {                       \
        new_shape[i] = rhs_shape[iter++];                                      \
      }                                                                        \
    }                                                                          \
    op_map[low] = builder::Reshape(op_map[low], new_shape);                    \
    if (op_map["X"].GetType().GetShape().size() == 4 &&                        \
        running_mode == RunningMode::ADAPTIVE) {                               \
      auto lv = builder::Transpose(op_map["X"], {0, 2, 3, 1});                 \
      auto rv = builder::Transpose(op_map["Y"], {0, 2, 3, 1});                 \
      auto res = builder::name(lv, rv);                                        \
      return std::make_shared<GcuOp>(                                          \
          builder::Transpose(builder::name(lv, rv), {0, 3, 1, 2}));            \
    }                                                                          \
    return std::make_shared<GcuOp>(builder::name(op_map["X"], op_map["Y"]));   \
  }

BINARY_OP(Add)
BINARY_OP(Sub)
BINARY_OP(Mul)
BINARY_OP(Div)
BINARY_OP(Greater)
BINARY_OP(LessEqual)
BINARY_OP(Less)
BINARY_OP(GreaterEqual)

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               ElementWisePowEquivalenceTrans) {
  std::vector<GcuOpPtr> inputs;
  if (map_inputs.count("X") != 0) {
    inputs.push_back(map_inputs["X"].at(0));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [X] gcu op"));
  }
  if (map_inputs.count("Y") != 0) {
    inputs.push_back(map_inputs["Y"].at(0));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Y] gcu op"));
  }
  auto *op = node->Op();
  auto lhs_shape = inputs[0]->GetType().GetShape();
  auto rhs_shape = inputs[1]->GetType().GetShape();
  print_shape(lhs_shape, "lhs");
  print_shape(rhs_shape, "rhs");
  if (lhs_shape == rhs_shape) {
    return std::make_shared<GcuOp>(builder::Pow(*inputs[0], *inputs[1]));
  }
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  std::cout << "---- axis: " << axis << std::endl;
  auto lhs_rank = inputs[0]->GetType().GetRank();
  auto rhs_rank = inputs[1]->GetType().GetRank();
  std::map<std::string, GcuOp> op_map{{"X", *inputs[0]}, {"Y", *inputs[1]}};
  auto low = lhs_rank < rhs_rank ? "X" : "Y";
  std::vector<int64_t> new_shape;
  int64_t iter = 0;
  if (lhs_rank < rhs_rank) {
    new_shape.assign(rhs_rank, 1);
    axis = axis > 0 ? axis : rhs_rank - lhs_rank;
    for (int64_t i = axis; i < axis + lhs_rank; ++i) {
      new_shape[i] = lhs_shape[iter++];
    }
  } else {
    new_shape.assign(lhs_rank, 1);
    axis = axis > 0 ? axis : lhs_rank - rhs_rank;
    for (int64_t i = axis; i < axis + rhs_rank; ++i) {
      new_shape[i] = rhs_shape[iter++];
    }
  }
  op_map[low] = builder::Reshape(op_map[low], new_shape);
  return std::make_shared<GcuOp>(builder::Pow(op_map["X"], op_map["Y"]));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AddGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  int32_t axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  if (axis == -1) {
    int32_t lhs_rank = lhs_op.GetType().GetShape().size();
    int32_t rhs_rank = rhs_op.GetType().GetShape().size();
    axis = std::abs(lhs_rank - rhs_rank);
  }

  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    if (lhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dx_shapes = lhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dx_axis = dx_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dx_rank = dx_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if (i < dx_axis || i >= dx_axis + dx_rank ||
            (dout_shapes[i] > 1 && dx_shapes[i - dx_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      lhs_grad_op = builder::ReduceSum(out_grad_op, false, reduce_axes);
      // if (lhs_grad_op.GetType().GetShape().size() == 0) {
      //   lhs_grad_op = builder::Reshape(lhs_grad_op, {1});
      // }
      if (dx_shapes != lhs_grad_op.GetType().GetShape()) {
        lhs_grad_op = builder::Reshape(lhs_grad_op, dx_shapes);
      }
    } else {
      lhs_grad_op = out_grad_op;
    }
  }

  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    if (rhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dy_shapes = rhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dy_rank = dy_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if ((i < dy_axis || i >= dy_axis + dy_rank) ||
            (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      rhs_grad_op = builder::ReduceSum(out_grad_op, false, reduce_axes);
      // if (rhs_grad_op.GetType().GetShape().size() == 0) {
      //   rhs_grad_op = builder::Reshape(rhs_grad_op, {1});
      // }
      if (dy_shapes != rhs_grad_op.GetType().GetShape()) {
        rhs_grad_op = builder::Reshape(rhs_grad_op, dy_shapes);
      }
    } else {
      rhs_grad_op = out_grad_op;
    }
  }

  if (lhs_grad_op.IsValid() && rhs_grad_op.IsValid()) {
    std::vector<GcuOp> outputs = {lhs_grad_op, rhs_grad_op};
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    std::vector<GcuPrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    tuple_dtype.push_back(lhs_grad_op.GetType().GetPrimitiveType());
    tuple_dtype.push_back(rhs_grad_op.GetType().GetPrimitiveType());
    tuple_shape.push_back(lhs_grad_op.GetType().GetShape());
    tuple_shape.push_back(rhs_grad_op.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (lhs_grad_op.IsValid()) {
    return std::make_shared<GcuOp>(lhs_grad_op);
  } else {
    return std::make_shared<GcuOp>(rhs_grad_op);
  }
}

// subgrad => x`: 1, y`: -1
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SubGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();

  int32_t axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  if (axis == -1) {
    int32_t lhs_rank = lhs_op.GetType().GetShape().size();
    int32_t rhs_rank = rhs_op.GetType().GetShape().size();
    axis = std::abs(lhs_rank - rhs_rank);
  }

  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    if (lhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dx_shapes = lhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dx_axis = dx_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dx_rank = dx_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if (i < dx_axis || i >= dx_axis + dx_rank ||
            (dout_shapes[i] > 1 && dx_shapes[i - dx_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      lhs_grad_op = builder::ReduceSum(out_grad_op, false, reduce_axes);
      if (dx_shapes != lhs_grad_op.GetType().GetShape()) {
        lhs_grad_op = builder::Reshape(lhs_grad_op, dx_shapes);
      }
    } else {
      lhs_grad_op = out_grad_op;
    }
  }

  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    auto tmp_out_grad_op = -out_grad_op;
    if (rhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dy_shapes = rhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dy_rank = dy_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if ((i < dy_axis || i >= dy_axis + dy_rank) ||
            (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      rhs_grad_op = builder::ReduceSum(tmp_out_grad_op, false, reduce_axes);
      if (dy_shapes != rhs_grad_op.GetType().GetShape()) {
        rhs_grad_op = builder::Reshape(rhs_grad_op, dy_shapes);
      }
    } else {
      rhs_grad_op = tmp_out_grad_op;
    }
  }

  if (lhs_grad_op.IsValid() && rhs_grad_op.IsValid()) {
    std::vector<GcuOp> outputs = {lhs_grad_op, rhs_grad_op};
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    std::vector<GcuPrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    tuple_dtype.push_back(lhs_grad_op.GetType().GetPrimitiveType());
    tuple_dtype.push_back(rhs_grad_op.GetType().GetPrimitiveType());
    tuple_shape.push_back(lhs_grad_op.GetType().GetShape());
    tuple_shape.push_back(rhs_grad_op.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (lhs_grad_op.IsValid()) {
    return std::make_shared<GcuOp>(lhs_grad_op);
  } else {
    return std::make_shared<GcuOp>(rhs_grad_op);
  }
}

// mulgrad => x`: y, y`: x
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MulGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));

  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto input_name_map = op->Inputs();

  int32_t axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  int32_t lhs_rank = lhs_op.GetType().GetShape().size();
  int32_t rhs_rank = rhs_op.GetType().GetShape().size();
  if (axis == -1) {
    axis = std::abs(lhs_rank - rhs_rank);
  }

  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    std::vector<int64_t> rb_dims(rhs_rank);
    std::iota(rb_dims.begin(), rb_dims.begin() + rhs_rank, axis);
    GcuOp rhs_did_op = rhs_op;
    if (rhs_rank != 0 && rhs_rank != lhs_rank) {
      rhs_did_op =
          builder::BroadcastInDim(rhs_op, rb_dims, out_grad_op.GetType());
    }
    lhs_grad_op = out_grad_op * rhs_did_op;
  }

  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    auto tmp_out_grad_op = out_grad_op * lhs_op;
    if (rhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dy_shapes = rhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dy_rank = dy_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if ((i < dy_axis || i >= dy_axis + dy_rank) ||
            (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      rhs_grad_op = builder::ReduceSum(tmp_out_grad_op, false, reduce_axes);
      if (dy_shapes != rhs_grad_op.GetType().GetShape()) {
        rhs_grad_op = builder::Reshape(rhs_grad_op, dy_shapes);
      }
    } else {
      rhs_grad_op = tmp_out_grad_op;
    }
  }

  if (lhs_grad_op.IsValid() && rhs_grad_op.IsValid()) {
    std::vector<GcuOp> outputs = {lhs_grad_op, rhs_grad_op};
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    std::vector<GcuPrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    tuple_dtype.push_back(lhs_grad_op.GetType().GetPrimitiveType());
    tuple_dtype.push_back(rhs_grad_op.GetType().GetPrimitiveType());
    tuple_shape.push_back(lhs_grad_op.GetType().GetShape());
    tuple_shape.push_back(rhs_grad_op.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (lhs_grad_op.IsValid()) {
    return std::make_shared<GcuOp>(lhs_grad_op);
  } else {
    return std::make_shared<GcuOp>(rhs_grad_op);
  }
}

// divgrad => x`: 1/y, y`: -x/y^2
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, DivGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto input_name_map = op->Inputs();

  int32_t axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  int32_t lhs_rank = lhs_op.GetType().GetShape().size();
  int32_t rhs_rank = rhs_op.GetType().GetShape().size();
  if (axis == -1) {
    axis = std::abs(lhs_rank - rhs_rank);
  }

  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  std::vector<int64_t> rb_dims(rhs_rank);
  std::iota(rb_dims.begin(), rb_dims.begin() + rhs_rank, axis);
  GcuOp rhs_did_op = rhs_op;
  if (rhs_rank != 0 && rhs_rank != lhs_rank) {
    rhs_did_op =
        builder::BroadcastInDim(rhs_op, rb_dims, out_grad_op.GetType());
  }

  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    lhs_grad_op = out_grad_op / rhs_did_op;
  }

  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    auto tmp_square = rhs_did_op * rhs_did_op;
    auto tmp_div = lhs_op / tmp_square;
    auto tmp_neg = -tmp_div;
    auto tmp_out_grad_op = out_grad_op * tmp_neg;
    if (rhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      std::vector<int64_t> reduce_axes;
      std::vector<int64_t> dst_shapes;
      std::vector<int64_t> dy_shapes = rhs_op.GetType().GetShape();
      std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
      int32_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
      int32_t dy_rank = dy_shapes.size();
      int32_t dout_rank = dout_shapes.size();
      for (int32_t i = 0; i < dout_rank; ++i) {
        if ((i < dy_axis || i >= dy_axis + dy_rank) ||
            (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
          reduce_axes.push_back(i);
        } else {
          dst_shapes.push_back(dout_shapes[i]);
        }
      }
      if (dst_shapes.size() == 0) {
        dst_shapes.push_back(1);
      }
      rhs_grad_op = builder::ReduceSum(tmp_out_grad_op, false, reduce_axes);
      if (dy_shapes != rhs_grad_op.GetType().GetShape()) {
        rhs_grad_op = builder::Reshape(rhs_grad_op, dy_shapes);
      }
    } else {
      rhs_grad_op = tmp_out_grad_op;
    }
  }

  if (lhs_grad_op.IsValid() && rhs_grad_op.IsValid()) {
    std::vector<GcuOp> outputs = {lhs_grad_op, rhs_grad_op};
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    std::vector<GcuPrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    tuple_dtype.push_back(lhs_grad_op.GetType().GetPrimitiveType());
    tuple_dtype.push_back(rhs_grad_op.GetType().GetPrimitiveType());
    tuple_shape.push_back(lhs_grad_op.GetType().GetShape());
    tuple_shape.push_back(rhs_grad_op.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (lhs_grad_op.IsValid()) {
    return std::make_shared<GcuOp>(lhs_grad_op);
  } else {
    return std::make_shared<GcuOp>(rhs_grad_op);
  }
}

// powgrad => x`: y * x^(y - 1), y`: lnx * x^y
IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               ElementWisePowGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto lhs_shape = lhs_op.GetType().GetShape();
  auto rhs_shape = rhs_op.GetType().GetShape();
  auto lhs_rank = lhs_op.GetType().GetRank();
  auto rhs_rank = rhs_op.GetType().GetRank();
  if (axis == -1) {
    axis = std::abs(lhs_rank - rhs_rank);
  }
  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  if (lhs_shape == rhs_shape) {
    if (output_name_map.count("X@GRAD") != 0 &&
        output_name_map["X@GRAD"].size() > 0) {
      GcuOp one = builder::OnesLike(rhs_op);
      lhs_grad_op = rhs_op * builder::Pow(lhs_op, rhs_op - one) * out_grad_op;
    }
    if (output_name_map.count("Y@GRAD") != 0 &&
        output_name_map["Y@GRAD"].size() > 0) {
      rhs_grad_op =
          builder::Pow(lhs_op, rhs_op) * builder::Log(lhs_op) * out_grad_op;
    }
  } else {
    std::vector<int64_t> new_shape;
    int64_t iter = 0;
    new_shape.assign(lhs_rank, 1);
    axis = axis > 0 ? axis : lhs_rank - rhs_rank;
    for (int64_t i = axis; i < axis + rhs_rank; ++i) {
      new_shape[i] = rhs_shape[iter++];
    }
    auto rhs_reshape_op = builder::Reshape(rhs_op, new_shape);
    if (output_name_map.count("X@GRAD") != 0 &&
        output_name_map["X@GRAD"].size() > 0) {
      GcuOp one = builder::OnesLike(rhs_reshape_op);
      lhs_grad_op = rhs_reshape_op *
                    builder::Pow(lhs_op, rhs_reshape_op - one) * out_grad_op;
    }
    if (output_name_map.count("Y@GRAD") != 0 &&
        output_name_map["Y@GRAD"].size() > 0) {
      rhs_grad_op = builder::Pow(lhs_op, rhs_reshape_op) *
                    builder::Log(lhs_op) * out_grad_op;
      if (rhs_shape != out_grad_op.GetType().GetShape()) {
        std::vector<int64_t> reduce_axes;
        std::vector<int64_t> dy_shapes = rhs_shape;
        std::vector<int64_t> dout_shapes = out_grad_op.GetType().GetShape();
        int64_t dy_axis = dy_shapes.size() < dout_shapes.size() ? axis : 0;
        int64_t dy_rank = dy_shapes.size();
        int64_t dout_rank = dout_shapes.size();
        for (int64_t i = 0; i < dout_rank; ++i) {
          if ((i < dy_axis || i >= dy_axis + dy_rank) ||
              (dout_shapes[i] > 1 && dy_shapes[i - dy_axis] == 1)) {
            reduce_axes.push_back(i);
          }
        }
        rhs_grad_op = builder::ReduceSum(rhs_grad_op, false, reduce_axes);
        if (dy_shapes != rhs_grad_op.GetType().GetShape()) {
          rhs_grad_op = builder::Reshape(rhs_grad_op, dy_shapes);
        }
      }
    }
  }
  if (lhs_grad_op.IsValid() && rhs_grad_op.IsValid()) {
    std::vector<GcuOp> outputs = {lhs_grad_op, rhs_grad_op};
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    std::vector<GcuPrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    tuple_dtype.push_back(lhs_grad_op.GetType().GetPrimitiveType());
    tuple_dtype.push_back(rhs_grad_op.GetType().GetPrimitiveType());
    tuple_shape.push_back(lhs_grad_op.GetType().GetShape());
    tuple_shape.push_back(rhs_grad_op.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (lhs_grad_op.IsValid()) {
    return std::make_shared<GcuOp>(lhs_grad_op);
  } else {
    return std::make_shared<GcuOp>(rhs_grad_op);
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kAdd, INSENSITIVE, AddEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kAddGrad, INSENSITIVE, AddGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSub, INSENSITIVE, SubEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSubGrad, INSENSITIVE, SubGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMul, INSENSITIVE, MulEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMulGrad, INSENSITIVE, MulGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kDiv, INSENSITIVE, DivEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kDivGrad, INSENSITIVE, DivGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGreater, INSENSITIVE, GreaterEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLessEqual, INSENSITIVE, LessEqualEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLessThan, INSENSITIVE, LessEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kElementWisePow,
                           INSENSITIVE,
                           ElementWisePowEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kElementWisePowGrad,
                           INSENSITIVE,
                           ElementWisePowGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGreaterEqual,
                           INSENSITIVE,
                           GreaterEqualEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
