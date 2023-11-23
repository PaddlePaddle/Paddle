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
const char *const kLabelSmooth = "label_smooth";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LabelSmoothEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  if (map_inputs.count("PriorDist") != 0) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "GCU doesn't support dist label smooth"));
  }
  auto ptype = input.GetType().GetPrimitiveType();
  auto scalar_type = builder::Type(ptype);
  auto input_shape = input.GetType().GetShape();

  int64_t cls_index = static_cast<int64_t>(input_shape.size() - 1);
  if (cls_index < 0) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "class number value should be greater than 0"));
  }
  float cls_k = 1.0 / input_shape[cls_index];
  auto prior_dist = builder::FullLike(input, cls_k);

  builder::Op eps_op;
  if (epsilon > 0.0) {
    eps_op = builder::FullLike(input, epsilon);
  } else {
    eps_op = builder::ZerosLike(input);
  }

  auto constant_ones = builder::OnesLike(input);
  auto result = (constant_ones - eps_op) * input + eps_op * prior_dist;
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kLabelSmooth,
                           INSENSITIVE,
                           LabelSmoothEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
