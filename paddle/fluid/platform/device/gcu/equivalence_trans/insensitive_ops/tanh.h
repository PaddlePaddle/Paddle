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
const char *const kTanh = "tanh";
const char *const kTanhGrad = "tanh_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TanhEquivalenceTrans) {
  builder::Op inputs = *(map_inputs["X"].at(0));
  auto scalar_type = builder::Type(inputs.GetType().GetPrimitiveType());
  float max_value = 13.0;
  float min_value = -40.0;
  auto max_op = builder::FullLike(inputs, max_value);
  auto min_op = builder::FullLike(inputs, min_value);
  inputs = builder::Clamp(min_op, inputs, max_op);
  return std::make_shared<GcuOp>(builder::Tanh(inputs));
}
// dy/dx=1-y2
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TanhGradEquivalenceTrans) {
  builder::Op out_op = *(map_inputs["Out"].at(0));
  builder::Op out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  return std::make_shared<GcuOp>(builder::TanhGrad(out_grad_op, out_op));
}
EQUIVALENCE_TRANS_FUNC_REG(kTanh, INSENSITIVE, TanhEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTanhGrad, INSENSITIVE, TanhGradEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
