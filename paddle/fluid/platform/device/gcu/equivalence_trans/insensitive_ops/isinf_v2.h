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

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kIsInfV2 = "isinf_v2";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, IsInfV2EquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  float inf_data = std::numeric_limits<float>::infinity();
  auto inf_op = builder::FullLike(input, inf_data);
  auto abs_in = builder::Abs(input);
  auto result = builder::Compare(abs_in, inf_op, "EQ");
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kIsInfV2, INSENSITIVE, IsInfV2EquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
