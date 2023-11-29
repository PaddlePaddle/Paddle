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
const char *const kReduceSum = "reduce_sum";
const char *const kReduceSumGrad = "reduce_sum_grad";
const char *const kReduceMean = "reduce_mean";
const char *const kReduceMeanGrad = "reduce_mean_grad";
const char *const kReduceMax = "reduce_max";
const char *const kReduceMin = "reduce_min";
const char *const kReduceProd = "reduce_prod";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReduceSumEquivalenceTrans) {
  auto *op = node->Op();
  auto x = *(map_inputs["X"].at(0));
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keep_dim = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis(dim.begin(), dim.end());
  auto input_rank = x.GetType().GetRank();
  if (reduce_all) {
    axis.assign(input_rank, 0);
    std::iota(axis.begin(), axis.end(), 0);
  }
  if (input_rank == 0) axis.clear();
  auto result = builder::ReduceSum(x, keep_dim, axis);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               ReduceSumGradEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op dout_op = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keepdims = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis;
  int64_t input_rank = x_op.GetType().GetRank();
  if (reduce_all) {
    if (input_rank > 0) {
      axis.assign(input_rank, 0);
      std::iota(axis.begin(), axis.end(), 0);
    }
  } else {
    for (auto value : dim) {
      if (value >= 0)
        axis.emplace_back(static_cast<int64_t>(value));
      else
        axis.emplace_back(static_cast<int64_t>(value) + input_rank);
    }
  }
  auto dx_op = dout_op;
  auto output_rank = dout_op.GetType().GetRank();
  if (keepdims) {
    auto output_shape = dout_op.GetType().GetShape();
    std::vector<int64_t> new_shape;
    size_t iter = 0;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (iter >= axis.size() || i != axis[iter]) {
        new_shape.emplace_back(output_shape[i]);
      } else {
        ++iter;
      }
    }
    dx_op = builder::Reshape(dx_op, new_shape);
  }
  std::vector<int64_t> broadcast_dims;
  size_t iter = 0;
  for (int64_t i = 0; i < input_rank; ++i) {
    if (iter >= axis.size() || i != axis[iter]) {
      broadcast_dims.emplace_back(i);
    } else {
      ++iter;
    }
  }
  broadcast_dims.resize(dx_op.GetType().GetRank());
  dx_op = builder::BroadcastInDim(dx_op, broadcast_dims, x_op.GetType());
  return std::make_shared<GcuOp>(dx_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReduceMeanEquivalenceTrans) {
  builder::Op x = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keep_dim = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis(dim.begin(), dim.end());
  auto input_rank = x.GetType().GetRank();
  if (reduce_all) {
    axis.assign(input_rank, 0);
    std::iota(axis.begin(), axis.end(), 0);
  }
  auto result = builder::ReduceMean(x, keep_dim, axis);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               ReduceMeanGradEquivalenceTrans) {
  auto *op = node->Op();
  builder::Op x = *(map_inputs["X"].at(0));
  builder::Op out = *(map_inputs["Out@GRAD"].at(0));
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  std::vector<int64_t> axis;
  auto input_rank = x.GetType().GetRank();
  if (!reduce_all) {
    for (auto &value : dim) {
      if (value < 0) value += input_rank;
      axis.emplace_back(static_cast<int64_t>(value));
    }
  } else {
    if (input_rank > 0) {
      axis.assign(input_rank, 0);
      std::iota(axis.begin(), axis.end(), 0);
    }
  }
  auto keepdims = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  auto output_size = out.GetType().GetSize();
  if (output_size == 0) {
    output_size = 1;
  }
  auto input_size = x.GetType().GetSize();
  if (input_size == 0) {
    input_size = 1;
  }
  float reduced_size = static_cast<float>(input_size / output_size);
  float reciprocal = 1.0 / reduced_size;
  builder::Op derivative = builder::FullLike(out, reciprocal);
  auto grad = out * derivative;
  auto output_rank = out.GetType().GetRank();
  if (keepdims) {
    auto output_shape = out.GetType().GetShape();
    std::vector<int64_t> new_shape;
    int iter = 0;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (i == axis[iter]) {
        ++iter;
      } else {
        new_shape.emplace_back(output_shape[i]);
      }
    }
    grad = builder::Reshape(grad, new_shape);
  }
  std::vector<int64_t> broadcast_dims;
  size_t iter = 0;
  for (int64_t i = 0; i < input_rank; ++i) {
    if (iter >= axis.size() || i != axis[iter]) {
      broadcast_dims.emplace_back(i);
    } else {
      ++iter;
    }
  }
  broadcast_dims.resize(grad.GetType().GetRank());
  auto result = builder::BroadcastInDim(grad, broadcast_dims, x.GetType());
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReduceMaxEquivalenceTrans) {
  auto *op = node->Op();
  auto x = *(map_inputs["X"].at(0));
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keep_dim = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis(dim.begin(), dim.end());
  auto input_rank = x.GetType().GetRank();
  if (reduce_all) {
    axis.assign(input_rank, 0);
    std::iota(axis.begin(), axis.end(), 0);
  }
  auto result = builder::ReduceMax(x, keep_dim, axis);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReduceMinEquivalenceTrans) {
  auto *op = node->Op();
  auto x = *(map_inputs["X"].at(0));
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keep_dim = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis(dim.begin(), dim.end());
  auto input_rank = x.GetType().GetRank();
  if (reduce_all) {
    axis.assign(input_rank, 0);
    std::iota(axis.begin(), axis.end(), 0);
  }
  auto result = builder::ReduceMin(x, keep_dim, axis);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReduceProdEquivalenceTrans) {
  auto *op = node->Op();
  auto x = *(map_inputs["X"].at(0));
  auto dim = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("dim"));
  auto reduce_all = PADDLE_GET_CONST(bool, op->GetAttr("reduce_all"));
  auto keep_dim = PADDLE_GET_CONST(bool, op->GetAttr("keep_dim"));
  std::vector<int64_t> axis(dim.begin(), dim.end());
  auto input_rank = x.GetType().GetRank();
  if (reduce_all) {
    axis.assign(input_rank, 0);
    std::iota(axis.begin(), axis.end(), 0);
  }
  auto result = builder::ReduceProd(x, keep_dim, axis);
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kReduceSum, INSENSITIVE, ReduceSumEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceSumGrad,
                           INSENSITIVE,
                           ReduceSumGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceMean,
                           INSENSITIVE,
                           ReduceMeanEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceMeanGrad,
                           INSENSITIVE,
                           ReduceMeanGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceMax, INSENSITIVE, ReduceMaxEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceMin, INSENSITIVE, ReduceMinEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReduceProd,
                           INSENSITIVE,
                           ReduceProdEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
