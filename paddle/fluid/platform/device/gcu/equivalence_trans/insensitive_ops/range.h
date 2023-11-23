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
const char *const kRange = "range";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RangeEquivalenceTrans) {
  GcuOp start = *(map_inputs["Start"].at(0));
  GcuOp end = *(map_inputs["End"].at(0));
  GcuOp step = *(map_inputs["Step"].at(0));
  auto start_rank = start.GetType().GetRank();
  auto end_rank = end.GetType().GetRank();
  auto step_rank = step.GetType().GetRank();
  PADDLE_ENFORCE(start_rank == 0 || start_rank == 1,
                 platform::errors::InvalidArgument(
                     "Range Op's start_rank must equal to 0 or 1, but got %d.",
                     start_rank));
  PADDLE_ENFORCE(
      end_rank == 0 || end_rank == 1,
      platform::errors::InvalidArgument(
          "Range Op's end_rank must equal to 0 or 1, but got %d.", end_rank));
  PADDLE_ENFORCE(
      step_rank == 0 || step_rank == 1,
      platform::errors::InvalidArgument(
          "Range Op's step_rank must equal to 0 or 1, but got %d.", step_rank));

  std::vector<int64_t> scalar_shape = {};

  start = builder::Reshape(start, scalar_shape);
  end = builder::Reshape(end, scalar_shape);
  step = builder::Reshape(step, scalar_shape);

  GcuOp result = builder::Range(start, end, step);
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kRange, INSENSITIVE, RangeEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
