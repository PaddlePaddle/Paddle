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
const char *const kAbs = "abs";
const char *const kAbsGrad = "abs_grad";
const char *const kExp = "exp";
const char *const kExpGrad = "exp_grad";
const char *const kReciprocal = "reciprocal";
const char *const kReciprocalGrad = "reciprocal_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AbsEquivalenceTrans) {
  return std::make_shared<GcuOp>(builder::Abs(*(map_inputs["X"].at(0))));
}

//           /  1, x > 0
// dy / dx = -  0, x = 0
//           \ -1, x < 0
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AbsGradEquivalenceTrans) {
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp zero = builder::ZerosLike(x);
  auto pred_negative = builder::Less(x, zero);
  auto temp = builder::Select(pred_negative, -out_grad, out_grad);
  auto pred_positive = builder::Equal(x, zero);
  return std::make_shared<GcuOp>(builder::Select(pred_positive, zero, temp));
}

// dy / dx = -(1/x)^2 = -y^2
IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               ReciprocalGradEquivalenceTrans) {
  GcuOp out_op = *(map_inputs["Out"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  GcuOp in_grad_op = -out_grad_op * out_op * out_op;
  return std::make_shared<GcuOp>(in_grad_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ExpEquivalenceTrans) {
  return std::make_shared<GcuOp>(builder::Exp(*(map_inputs["X"].at(0))));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ExpGradEquivalenceTrans) {
  GcuOp out = *(map_inputs["Out"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  return std::make_shared<GcuOp>(out * out_grad);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReciprocalEquivalenceTrans) {
  return std::make_shared<GcuOp>(builder::Reciprocal(*(map_inputs["X"].at(0))));
}

EQUIVALENCE_TRANS_FUNC_REG(kAbs, INSENSITIVE, AbsEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kAbsGrad, INSENSITIVE, AbsGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kExp, INSENSITIVE, ExpEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kExpGrad, INSENSITIVE, ExpGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReciprocal,
                           INSENSITIVE,
                           ReciprocalEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReciprocalGrad,
                           INSENSITIVE,
                           ReciprocalGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
