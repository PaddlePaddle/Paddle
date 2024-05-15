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
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kMean = "mean";
const char *const kMeanGrad = "mean_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MeanEquivalenceTrans) {
  builder::Op inputs = *(map_inputs["X"].at(0));
  auto ptype = inputs.GetType().GetPrimitiveType();
  auto input_shape = inputs.GetType().GetShape();
  bool keepdims = false;
  int64_t dim = static_cast<int64_t>(input_shape.size());
  auto type = builder::Type(ptype);
  std::vector<int64_t> axis;
  for (int64_t i = 0; i < dim; i++) {
    axis.emplace_back(i);
  }
  auto result = builder::ReduceMean(inputs, keepdims, axis, type);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MeanGradEquivalenceTrans) {
  builder::Op x = *(map_inputs["X"].at(0));
  builder::Op out = *(map_inputs["Out@GRAD"].at(0));
  int64_t dim = static_cast<int64_t>(x.GetType().GetShape().size());
  std::vector<int64_t> axis;
  for (int64_t i = 0; i < dim; i++) {
    axis.emplace_back(i);
  }
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
  std::vector<int64_t> broadcast_dims;
  int iter = 0;
  for (int64_t i = 0; i < output_rank; ++i) {
    if (i == axis[iter]) {
      ++iter;
    } else {
      broadcast_dims.emplace_back(i);
    }
  }
  auto result = builder::BroadcastInDim(grad, broadcast_dims, x.GetType());
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kMean, INSENSITIVE, MeanEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMeanGrad, INSENSITIVE, MeanGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
