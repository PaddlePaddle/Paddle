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
const char *const kScale = "scale";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ScaleEquivalenceTrans) {
  auto input = map_inputs["X"].at(0);
  PADDLE_ENFORCE_NOT_NULL(input);
  auto *op = node->Op();
  auto scale = PADDLE_GET_CONST(float, op->GetAttr("scale"));
  auto bias = PADDLE_GET_CONST(float, op->GetAttr("bias"));
  auto bias_after_scale =
      PADDLE_GET_CONST(bool, op->GetAttr("bias_after_scale"));
  if (map_inputs.count("bias") != 0) {
    auto scale_op = map_inputs["ScaleTensor"].at(0);
    auto bias_op = map_inputs["bias"].at(0);
    PADDLE_ENFORCE_NOT_NULL(scale_op);
    PADDLE_ENFORCE_NOT_NULL(bias_op);
    // cast process
    auto dtype = input->GetType().GetPrimitiveType();
    if (dtype != builder::PrimitiveType::F32()) {
      if (SizeOf(dtype) <= 4) {
        builder::Type output_type(input->GetType().GetShape(),
                                  builder::PrimitiveType::F32());
        input = std::make_shared<builder::Op>(
            builder::Convert(*input, output_type));
      } else {
        // convert to F64
        builder::Type output_type(input->GetType().GetShape(),
                                  builder::PrimitiveType::F64());
        builder::Type scale_type({}, builder::PrimitiveType::F64());
        builder::Type bias_type({}, builder::PrimitiveType::F64());
        input = std::make_shared<builder::Op>(
            builder::Convert(*input, output_type));
        scale_op = std::make_shared<builder::Op>(
            builder::Convert(*scale_op, scale_type));
        bias_op = std::make_shared<builder::Op>(
            builder::Convert(*bias_op, bias_type));
      }
    }

    builder::Type res_type(input->GetType().GetShape(), dtype);
    if (bias_after_scale) {
      auto res = (*input) * (*scale_op) + (*bias_op);
      return std::make_shared<GcuOp>(builder::Convert(res, res_type));
    } else {
      auto res = ((*input) + (*bias_op)) * (*scale_op);
      return std::make_shared<GcuOp>(builder::Convert(res, res_type));
    }
  } else {
    builder::Op scale_op;
    if (map_inputs.count("ScaleTensor") != 0) {
      scale_op = *(map_inputs["ScaleTensor"].at(0));
    }
    if (!scale_op.IsValid()) {
      scale_op = builder::FullLike(*input, scale);
    }
    auto bias_op = builder::FullLike(*input, bias);
    if (bias_after_scale) {
      return std::make_shared<GcuOp>((*input) * scale_op + bias_op);
    } else {
      return std::make_shared<GcuOp>(((*input) + bias_op) * scale_op);
    }
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kScale, INSENSITIVE, ScaleEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
