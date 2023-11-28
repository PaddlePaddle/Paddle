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
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {

const char* const kSigmoidCrossEntropyWithLogits =
    "sigmoid_cross_entropy_with_logits";
const char* const kSigmoidCrossEntropyWithLogitsGrad =
    "sigmoid_cross_entropy_with_logits_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               SigmoidCrossEntropyWithLogitsEquivalenceTrans) {
  GcuOp logits_op = *(map_inputs["X"].at(0));
  GcuOp label_op = *(map_inputs["Label"].at(0));
  auto* op = node->Op();
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));
  auto normalize = PADDLE_GET_CONST(bool, op->GetAttr("normalize"));
  auto ptype = logits_op.GetType().GetPrimitiveType();
  float zero_data = 0.0;
  GcuOp zero = builder::Const(
      gcu_builder, static_cast<void*>(&zero_data), builder::Type(ptype));
  float one_data = 1.0;
  GcuOp one = builder::Const(
      gcu_builder, static_cast<void*>(&one_data), builder::Type(ptype));
  GcuOp logits_max = builder::Max(logits_op, zero);
  GcuOp logits_abs = builder::Abs(logits_op);
  GcuOp logits_exp = builder::Exp(-logits_abs);
  GcuOp logits_log = builder::Log(one + logits_exp);
  float ignore_index_data = static_cast<float>(ignore_index);
  GcuOp ignore_index_op = builder::Const(gcu_builder,
                                         static_cast<void*>(&ignore_index_data),
                                         builder::Type(ptype));
  GcuOp ignore_index_broadcast =
      builder::BroadcastInDim(ignore_index_op, {}, label_op.GetType());
  GcuOp pred = builder::Equal(label_op, ignore_index_broadcast);
  GcuOp label_select = builder::Select(pred, zero, one, label_op.GetType());
  GcuOp output = logits_max - logits_op * label_op + logits_log;
  if (normalize) {
    GcuOp ignore_select = builder::Select(pred, zero, one, label_op.GetType());
    GcuOp ignore_sum = builder::ReduceSum(ignore_select, false);
    return std::make_shared<GcuOp>(output * label_select / ignore_sum);
  }
  return std::make_shared<GcuOp>(output * label_select);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder,
    node,
    map_inputs,
    running_mode,
    SigmoidCrossEntropyWithLogitsGradEquivalenceTrans) {
  GcuOp logits_op = *(map_inputs["X"].at(0));
  GcuOp label_op = *(map_inputs["Label"].at(0));
  GcuOp out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  auto ptype = logits_op.GetType().GetPrimitiveType();
  float zero_data = 0.0;
  GcuOp zero = builder::Const(
      gcu_builder, static_cast<void*>(&zero_data), builder::Type(ptype));
  float one_data = 1.0;
  GcuOp one = builder::Const(
      gcu_builder, static_cast<void*>(&one_data), builder::Type(ptype));
  GcuOp pred = builder::Greater(logits_op, zero);
  GcuOp logits_abs = builder::Abs(logits_op);
  GcuOp logits_exp = builder::Exp(-logits_abs);
  GcuOp logits_max_grad = builder::Select(pred, one, zero, logits_op.GetType());
  GcuOp logits_abs_grad = builder::Select(pred, -one, one, logits_op.GetType());
  GcuOp logits_log_grad = one / (one + logits_exp);
  GcuOp output = out_grad_op * ((logits_max_grad - label_op) +
                                logits_log_grad * logits_exp * logits_abs_grad);
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kSigmoidCrossEntropyWithLogits,
                           INSENSITIVE,
                           SigmoidCrossEntropyWithLogitsEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSigmoidCrossEntropyWithLogitsGrad,
                           INSENSITIVE,
                           SigmoidCrossEntropyWithLogitsGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
