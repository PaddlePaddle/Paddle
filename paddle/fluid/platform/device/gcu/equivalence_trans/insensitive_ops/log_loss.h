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
const char *const kLogLoss = "log_loss";
const char *const kLogLossGrad = "log_loss_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LogLossEquivalenceTrans) {
  auto *op = node->Op();
  double epsilon;
  auto eps_attr = op->GetAttr("epsilon");
  if (eps_attr.type() == typeid(float)) {
    epsilon = PADDLE_GET_CONST(float, eps_attr);
  } else if (eps_attr.type() == typeid(double)) {
    epsilon = PADDLE_GET_CONST(double, eps_attr);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported data type for log_loss epsilon attribute."));
  }
  auto predicted_op = *(map_inputs["Predicted"].at(0));
  auto labels_op = *(map_inputs["Labels"].at(0));

  auto eps_op = builder::FullLike(predicted_op, epsilon);
  auto const_1_op = builder::OnesLike(predicted_op);

  auto out_op = (-labels_op) * builder::Log(predicted_op + eps_op) -
                (const_1_op - labels_op) *
                    builder::Log(const_1_op - predicted_op + eps_op);

  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LogLossGradEquivalenceTrans) {
  auto *op = node->Op();
  double epsilon;
  auto eps_attr = op->GetAttr("epsilon");
  if (eps_attr.type() == typeid(float)) {
    epsilon = PADDLE_GET_CONST(float, eps_attr);
  } else if (eps_attr.type() == typeid(double)) {
    epsilon = PADDLE_GET_CONST(double, eps_attr);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unsupported data type for log_loss epsilon attribute."));
  }
  auto predicted_op = *(map_inputs["Predicted"].at(0));
  auto labels_op = *(map_inputs["Labels"].at(0));
  auto loss_grad_op = *(map_inputs["Loss@GRAD"].at(0));

  auto eps_op = builder::FullLike(predicted_op, epsilon);
  auto const_1_op = builder::OnesLike(predicted_op);

  auto pred_grad_op =
      loss_grad_op *
      (-(labels_op / (predicted_op + eps_op)) +
       ((const_1_op - labels_op) / (const_1_op - predicted_op + eps_op)));

  return std::make_shared<GcuOp>(pred_grad_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kLogLoss, INSENSITIVE, LogLossEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLogLossGrad,
                           INSENSITIVE,
                           LogLossGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
