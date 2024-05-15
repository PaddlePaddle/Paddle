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
const char *const kFlip = "flip";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FlipEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axis"));
  auto rank = input.GetType().GetRank();
  std::vector<int64_t> new_axis;
  for (size_t i = 0; i < axis.size(); i++) {
    if (axis[i] < 0) axis[i] += rank;
    new_axis.emplace_back(static_cast<int64_t>(axis[i]));
  }
  auto output = builder::Reverse(input, new_axis);
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kFlip, INSENSITIVE, FlipEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
