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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kMinimum = "elementwise_min";
const char *const kMinimumGrad = "elementwise_min_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MinimumEquivalenceTrans) {
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

  if (lhs_shape == rhs_shape) {
    return std::make_shared<GcuOp>(builder::Min(*inputs[0], *inputs[1]));
  }

  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
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
  return std::make_shared<GcuOp>(builder::Min(op_map["X"], op_map["Y"]));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MinimumGradEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp lhs = *(map_inputs["X"].at(0));
  GcuOp rhs = *(map_inputs["Y"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto output_name_map = op->Outputs();

  bool grad_x = output_name_map.count("X@GRAD") != 0 &&
                output_name_map["X@GRAD"].size() > 0;
  bool grad_y = output_name_map.count("Y@GRAD") != 0 &&
                output_name_map["Y@GRAD"].size() > 0;

  int32_t lhs_rank = lhs.GetType().GetShape().size();
  int32_t rhs_rank = rhs.GetType().GetShape().size();
  int32_t rank_diff = std::abs(lhs_rank - rhs_rank);

  int flag = 0;
  if (lhs_rank != rhs_rank) {
    std::vector<int64_t> broadcast_dimensions;
    const int64_t max_dim = std::max(lhs_rank, rhs_rank);
    const int64_t min_dim = std::min(lhs_rank, rhs_rank);
    broadcast_dimensions.resize(min_dim);
    std::iota(broadcast_dimensions.begin(),
              broadcast_dimensions.end(),
              max_dim - min_dim);
    if (lhs_rank > rhs_rank) {
      if (rhs_rank == 1 && rhs.GetType().GetShape()[0] == 1)
        rhs = builder::BroadcastInDim(rhs, {}, lhs.GetType());
      else
        rhs = builder::BroadcastInDim(rhs, broadcast_dimensions, lhs.GetType());
    } else {
      if (lhs_rank == 1 && lhs.GetType().GetShape()[0] == 1)
        lhs = builder::BroadcastInDim(lhs, {}, rhs.GetType());
      else
        lhs = builder::BroadcastInDim(lhs, broadcast_dimensions, rhs.GetType());
    }
  }

  builder::Type type = out_grad.GetType();
  auto ptype = type.GetPrimitiveType();
  auto shape = type.GetShape();
  builder::Type scalar_type(ptype);
  builder::Type pred_type(shape, builder::PrimitiveType::PRED());
  auto pred = builder::Compare(lhs, rhs, "LT", {}, pred_type);
  std::vector<float> zero_data{0};
  void *data_ptr = static_cast<void *>(zero_data.data());
  auto zero = builder::Const(gcu_builder, data_ptr, scalar_type);
  auto zeros = builder::ZerosLike(out_grad);
  builder::Op lhs_grad;
  builder::Op rhs_grad;
  std::vector<builder::PrimitiveType> tuple_dtype(2, ptype);
  std::vector<std::vector<int64_t>> tuple_shape;

  if (grad_x && grad_y) {
    auto lhs_grad = builder::Select(pred, out_grad, zeros, type);
    auto rhs_grad = builder::Select(pred, zeros, out_grad, type);

    if (lhs_rank > rhs_rank) {
      std::vector<int64_t> axis(rank_diff);
      std::iota(axis.begin(), axis.end(), 0);
      rhs_grad = builder::ReduceSum(rhs_grad, false, axis);

    } else if (lhs_rank < rhs_rank) {
      std::vector<int64_t> axis(rank_diff);
      std::iota(axis.begin(), axis.end(), 0);
      lhs_grad = builder::ReduceSum(lhs_grad, false, axis);
    }

    tuple_shape.emplace_back(lhs_grad.GetType().GetShape());
    tuple_shape.emplace_back(rhs_grad.GetType().GetShape());
    builder::Type outputs_type(tuple_shape, tuple_dtype);

    std::vector<std::string> output_names{"X@GRAD", "Y@GRAD"};
    std::string output_names_attr(output_name_map[output_names[0]][0]);
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    auto res = builder::Tuple({lhs_grad, rhs_grad}, outputs_type);
    res.SetAttribute(kAttrOpOutVarName,
                     builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<builder::Op>(res);

  } else if (grad_x) {
    auto lhs_grad = builder::Select(pred, out_grad, zeros, type);
    if (flag == 2) {
      std::vector<int64_t> axis(shape.size());
      std::iota(axis.begin(), axis.end(), 0);
      lhs_grad = builder::ReduceSum(lhs_grad, false, axis);
      if (lhs_rank == 0) {
        lhs_grad = builder::Reshape(lhs_grad, scalar_type);
      }
    }
    return std::make_shared<builder::Op>(lhs_grad);
  } else {
    auto rhs_grad = builder::Select(pred, zeros, out_grad, type);
    if (flag == 1) {
      std::vector<int64_t> axis(shape.size());
      std::iota(axis.begin(), axis.end(), 0);
      rhs_grad = builder::ReduceSum(rhs_grad, false, axis);
      if (rhs_rank == 0) {
        rhs_grad = builder::Reshape(rhs_grad, scalar_type);
      }
    }
    return std::make_shared<builder::Op>(rhs_grad);
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kMinimum, INSENSITIVE, MinimumEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMinimumGrad,
                           INSENSITIVE,
                           MinimumGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
