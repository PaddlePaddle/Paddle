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
const char *const kNotEqual = "not_equal";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, kNotEqualEquivalenceTrans) {
  builder::Op x_op = *(map_inputs["X"].at(0));
  builder::Op y_op = *(map_inputs["Y"].at(0));
  return std::make_shared<GcuOp>(builder::NotEqual(x_op, y_op));
}

EQUIVALENCE_TRANS_FUNC_REG(kNotEqual, INSENSITIVE, kNotEqualEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
