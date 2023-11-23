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
const char *const kArgMax = "arg_max";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ArgMaxEquivalenceTrans) {
  auto *op = node->Op();
  int64_t axis = PADDLE_GET_CONST(int64_t, op->GetAttr("axis"));
  auto keepdims = PADDLE_GET_CONST(bool, op->GetAttr("keepdims"));
  auto flatten = PADDLE_GET_CONST(bool, op->GetAttr("flatten"));
  GcuOp data = *(map_inputs["X"].at(0));
  int64_t rank = data.GetType().GetRank();
  GcuOp result;
  if (flatten) {
    auto data_shape = data.GetType().GetShape();
    int64_t new_shape = 1;
    for (auto dim : data_shape) {
      new_shape *= dim;
    }
    builder::Type output_type({new_shape}, data.GetType().GetPrimitiveType());
    auto out = builder::Reshape(data, output_type);
    result = builder::ArgMax(out, /*axis=*/0, keepdims);
  } else {
    if (axis < 0) {
      axis = axis + rank;
    }
    result = builder::ArgMax(data, axis, keepdims);
  }
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kArgMax, INSENSITIVE, ArgMaxEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
