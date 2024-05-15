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
const char *const kHuberLoss = "huber_loss";
const char *const kHuberLossGrad = "huber_loss_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, HuberLossEquivalenceTrans) {
  auto *op = node->Op();
  float delta = PADDLE_GET_CONST(float, op->GetAttr("delta"));
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp y = *(map_inputs["Y"].at(0));
  GcuOp residual = y - x;
  GcuOp residual_abs = builder::Abs(residual);
  GcuOp delta_op = builder::FullLike(residual, delta);
  GcuOp scale_op = builder::FullLike(residual, 0.5f);  // scale = 0.5f
  GcuOp pred = builder::Greater(residual_abs, delta_op);
  GcuOp greater_d = delta_op * (residual_abs - scale_op * delta_op);
  GcuOp less_d = scale_op * (residual_abs * residual_abs);
  GcuOp output = builder::Select(pred, greater_d, less_d);
  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["Residual"].at(0) + ";";
  output_names_attr += output_name_map["Out"].at(0);
  auto result = builder::Tuple({residual, output});
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               HuberLossGradEquivalenceTrans) {
  auto *op = node->Op();
  float delta = PADDLE_GET_CONST(float, op->GetAttr("delta"));
  GcuOp residual = *(map_inputs["Residual"].at(0));
  GcuOp dout = *(map_inputs["Out@GRAD"].at(0));
  GcuOp delta_op = builder::FullLike(residual, delta);
  GcuOp residual_abs = builder::Abs(residual);
  GcuOp pred = builder::Greater(residual_abs, delta_op);
  GcuOp pred_neg = builder::Less(residual, -delta_op);
  GcuOp dx = builder::Select(pred, -delta_op, -residual) * dout;
  dx = builder::Select(pred_neg, -dx, dx);

  GcuOp dy = builder::Select(pred, delta_op, residual) * dout;
  dy = builder::Select(pred_neg, -dy, dy);
  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["X@GRAD"].at(0) + ";";
  auto result = builder::Tuple({dx});
  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() != 0) {
    output_names_attr += ";" + output_name_map["Y@GRAD"][0];
    result = builder::Tuple({dx, dy});
  }
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kHuberLoss, INSENSITIVE, HuberLossEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kHuberLossGrad,
                           INSENSITIVE,
                           HuberLossGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
