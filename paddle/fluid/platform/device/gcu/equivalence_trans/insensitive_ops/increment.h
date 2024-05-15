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
const char *const kIncrement = "increment";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, IncrementEquivalenceTrans) {
  auto *op = node->Op();
  assert(op != nullptr);
  builder::Op inputs = *(map_inputs["X"].at(0));
  auto step = PADDLE_GET_CONST(float, op->GetAttr("step"));
  auto input_shape = inputs.GetType().GetShape();
  if (input_shape.size() != 1) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support tensor of one element"));
  }
  auto value_op = builder::FullLike(inputs, step);
  return std::make_shared<GcuOp>(inputs + value_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kIncrement, INSENSITIVE, IncrementEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
