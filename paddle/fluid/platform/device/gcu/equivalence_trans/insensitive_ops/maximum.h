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

#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kMaximum = "elementwise_max";
const char *const kMaximumGrad = "elementwise_max_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MaximumEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp output = builder::Max(lhs_op, rhs_op);
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MaximumGradEquivalenceTrans) {
  GcuOp lhs_op = *(map_inputs["X"].at(0));
  GcuOp rhs_op = *(map_inputs["Y"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  GcuOp output = builder::Max(lhs_op, rhs_op);
  std::vector<int64_t> lhs_shapes = lhs_op.GetType().GetShape();
  std::vector<int64_t> rhs_shapes = rhs_op.GetType().GetShape();
  std::vector<int64_t> reduce_dims;
  std::vector<int64_t> reshape_dim;
  int64_t lhs_rank = lhs_shapes.size();
  int64_t rhs_rank = rhs_shapes.size();
  GcuOp lhs_broadcast, rhs_broadcast;
  if (lhs_shapes.size() > rhs_shapes.size()) {
    for (int64_t i = 0; i < lhs_rank; ++i) {
      if (i < rhs_rank - 1) {
        reduce_dims.push_back(i);
      }
      if (i < lhs_rank - rhs_rank) {
        reshape_dim.push_back(1);
      } else {
        reshape_dim.push_back(lhs_shapes[i]);
      }
    }
    std::vector<int64_t> broadcast_dims(lhs_rank);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    auto rhs_reshape = builder::Reshape(rhs_op, reshape_dim);
    rhs_broadcast =
        builder::BroadcastInDim(rhs_reshape, broadcast_dims, lhs_op.GetType());
  } else {
    for (int64_t i = 0; i < rhs_rank; ++i) {
      if (i < lhs_rank - 1) {
        reduce_dims.push_back(i);
      }
      if (i < rhs_rank - lhs_rank) {
        reshape_dim.push_back(1);
      } else {
        reshape_dim.push_back(rhs_shapes[i]);
      }
    }
    std::vector<int64_t> broadcast_dims(rhs_rank);
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
    auto lhs_reshape = builder::Reshape(lhs_op, reshape_dim);
    lhs_broadcast =
        builder::BroadcastInDim(lhs_reshape, broadcast_dims, rhs_op.GetType());
  }

  GcuOp lhs_pred;
  GcuOp rhs_pred;
  if (rhs_broadcast.IsValid()) {
    lhs_pred = builder::Greater(lhs_op, rhs_broadcast);
  }
  if (lhs_broadcast.IsValid()) {
    lhs_pred = builder::Greater(lhs_broadcast, rhs_op);
  }
  if (lhs_broadcast.IsValid()) {
    rhs_pred = builder::Equal(output, rhs_op);
  }
  if (rhs_broadcast.IsValid()) {
    rhs_pred = builder::Equal(output, rhs_broadcast);
  }
  GcuOp lhs_select, rhs_select;
  auto zero = builder::ZerosLike(lhs_op);
  auto one = builder::OnesLike(lhs_op);
  lhs_select = builder::Select(lhs_pred, one, zero, out_grad_op.GetType());
  rhs_select = builder::Select(rhs_pred, one, zero, out_grad_op.GetType());
  GcuOp lhs_grad_op, rhs_grad_op;
  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    lhs_grad_op = lhs_select * out_grad_op;
    if (lhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      lhs_grad_op = builder::ReduceSum(lhs_grad_op, false, reduce_dims);
    }
  }
  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    rhs_grad_op = rhs_select * out_grad_op;
    if (rhs_op.GetType().GetShape() != out_grad_op.GetType().GetShape()) {
      rhs_grad_op = builder::ReduceSum(rhs_grad_op, false, reduce_dims);
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

EQUIVALENCE_TRANS_FUNC_REG(kMaximum, INSENSITIVE, MaximumEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMaximumGrad,
                           INSENSITIVE,
                           MaximumGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
