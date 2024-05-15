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
const char *const kClip = "clip";
const char *const kClipGrad = "clip_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ClipEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto op = node->Op();
  auto min = static_cast<float>(PADDLE_GET_CONST(float, op->GetAttr("min")));
  auto max = static_cast<float>(PADDLE_GET_CONST(float, op->GetAttr("max")));

  auto min_op = builder::FullLike(input, min);
  auto max_op = builder::FullLike(input, max);

  auto output = builder::Clamp(min_op, input, max_op, input.GetType());
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ClipGradEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto op = node->Op();
  auto min = static_cast<float>(PADDLE_GET_CONST(float, op->GetAttr("min")));
  auto max = static_cast<float>(PADDLE_GET_CONST(float, op->GetAttr("max")));

  auto min_op = builder::FullLike(input, min);
  auto max_op = builder::FullLike(input, max);

  auto output = builder::ClampGrad(out_grad, input, min_op, max_op);
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kClip, INSENSITIVE, ClipEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kClipGrad, INSENSITIVE, ClipGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
