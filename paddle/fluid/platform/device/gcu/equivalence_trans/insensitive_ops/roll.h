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

#include <iostream>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kRoll = "roll";
const char *const kRollGrad = "roll_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RollEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto shifts = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shifts"));
  auto axis = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("axis"));
  auto output = builder::Roll(input, shifts, axis);
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RollGradEquivalenceTrans) {
  builder::Op grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto shifts = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shifts"));
  auto axis = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("axis"));
  auto grad_in = builder::RollGrad(grad_out, shifts, axis);
  return std::make_shared<GcuOp>(grad_in);
}

EQUIVALENCE_TRANS_FUNC_REG(kRoll, INSENSITIVE, RollEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRollGrad, INSENSITIVE, RollGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
