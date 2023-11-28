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
#include <algorithm>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kCumsum = "cumsum";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, CumsumEquivalenceTrans) {
  auto *op = node->Op();
  auto input = *(map_inputs["X"].at(0));
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto flatten = PADDLE_GET_CONST(bool, op->GetAttr("flatten"));
  auto exclusive = PADDLE_GET_CONST(bool, op->GetAttr("exclusive"));
  auto reverse = PADDLE_GET_CONST(bool, op->GetAttr("reverse"));

  if (flatten) {
    input = builder::FlattenV2(input);
    axis = 0;
  }

  auto axis_op =
      builder::Const(input.GetBuilder(),
                     static_cast<void *>(&axis),
                     builder::Type({1}, builder::PrimitiveType::S32()));

  auto out_op = builder::CumSum(input, axis_op, exclusive, reverse);
  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kCumsum, INSENSITIVE, CumsumEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
