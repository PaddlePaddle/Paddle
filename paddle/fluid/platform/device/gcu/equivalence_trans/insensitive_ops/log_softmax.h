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
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kLogSoftmax = "log_softmax";
const char *const kLogSoftmaxGrad = "log_softmax_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LogSoftmaxEquivalenceTrans) {
  auto *op = node->Op();
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  auto input = *(map_inputs["X"].at(0));
  auto log_softmax_results = builder::Softmax(input, axis, true, true, 0.0);
  return std::make_shared<GcuOp>(log_softmax_results);
}

static int64_t GetCanonicalDimensionIndex(int64_t dim, int64_t rank) {
  int64_t min_shape_dim = -rank;
  int64_t max_shape_dim = rank - 1;
  if (!(min_shape_dim <= dim && dim <= max_shape_dim)) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Value out of range (expected to be in range of [%d, %d], but got %d)",
        min_shape_dim,
        max_shape_dim,
        dim));
  }
  int64_t dim_index = dim < 0 ? rank + dim : dim;
  if (dim_index < 0 || dim_index >= rank) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Value out of dim_index expected to be lower than 0 and equal or "
        "larger than rank"));
  }
  return dim_index;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               LogSoftmaxGradEquivalenceTrans) {
  auto SoftmaxSumOfGrad = [](builder::Op grad_output,
                             int64_t dim) -> builder::Op {
    return builder::ReduceSum(grad_output, false, {dim});
  };
  auto *op = node->Op();
  int64_t axis =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));
  builder::Op out = *(map_inputs["Out"].at(0));
  if (axis < 0) axis += out.GetType().GetRank();

  auto type = out_grad.GetType().GetPrimitiveType();
  auto rank = out_grad.GetType().GetShape().size();
  int64_t canonical_dim = GetCanonicalDimensionIndex(axis, rank);
  builder::Op sum = SoftmaxSumOfGrad(out_grad, canonical_dim);
  auto broadcast_dimensions = GetBroadcastDimensions(rank, canonical_dim);
  sum = builder::BroadcastInDim(
      sum,
      broadcast_dimensions,
      {out.GetType().GetShape(), sum.GetType().GetPrimitiveType()});
  builder::Op result_op =
      builder::Sub(out_grad, builder::Mul(builder::Exp(out), sum));
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kLogSoftmax,
                           INSENSITIVE,
                           LogSoftmaxEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLogSoftmaxGrad,
                           INSENSITIVE,
                           LogSoftmaxGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
