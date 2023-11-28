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
const char *const kDropout = "dropout";
const char *const kDropoutGrad = "dropout_grad";
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, DropoutEquivalenceTrans) {
  auto *op = node->Op();
  auto prob = PADDLE_GET_CONST(float, op->GetAttr("dropout_prob"));
  auto prob_1 = 1 - prob;
  auto dropout_imple =
      PADDLE_GET_CONST(std::string, op->GetAttr("dropout_implementation"));
  auto sed = PADDLE_GET_CONST(int, op->GetAttr("seed"));
  auto is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  int64_t seed = static_cast<int64_t>(sed);
  builder::Op input = *(map_inputs["X"].at(0));
  builder::Op prob1_op = builder::FullLike(input, prob_1);
  builder::Op mask;
  builder::Op output;
  auto zero = builder::ZerosLike(input);
  auto one = builder::OnesLike(input);
  if (is_test) {
    if (dropout_imple == "downgrade_in_infer") {
      output = input * prob1_op;
    } else {
      output = input;
    }
    mask = builder::Convert(
        one, {input.GetType().GetShape(), builder::PrimitiveType::S8()});
  } else {
    if (prob == 1.0f) {
      mask = zero;
      output = input * mask;
      mask = builder::Convert(
          mask, {input.GetType().GetShape(), builder::PrimitiveType::S8()});
    } else {
      if (dropout_imple == "upscale_in_train") {
        input = input / prob1_op;
      }
      std::vector<int64_t> shape_data = input.GetType().GetShape();
      std::vector<int32_t> shape_data_32;
      for (size_t i = 0; i < shape_data.size(); ++i) {
        shape_data_32.push_back(static_cast<int32_t>(shape_data[i]));
      }
      int32_t shape_dims = static_cast<int32_t>(shape_data.size());
      auto shape_op =
          builder::Const(gcu_builder,
                         static_cast<void *>(shape_data_32.data()),
                         {{shape_dims}, builder::PrimitiveType::S32()});
      auto scalar_zero =
          builder::ZerosLike(input, input.GetType().GetPrimitiveType(), {1});
      auto scalar_one =
          builder::OnesLike(input, input.GetType().GetPrimitiveType(), {1});
      auto mask_rnd =
          builder::RngUniform(scalar_zero, scalar_one, shape_op, seed);
      std::vector<int64_t> broadcast_dims(input.GetType().GetRank());
      std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
      auto prob_op = builder::FullLike(input, prob);
      auto pred = builder::Greater(mask_rnd, prob_op);
      mask = builder::Select(pred, one, zero);
      output = input * mask;
      mask = builder::Convert(
          mask, {mask.GetType().GetShape(), builder::PrimitiveType::S8()});
    }
  }
  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["Out"].at(0) + ";";
  output_names_attr += output_name_map["Mask"].at(0);

  auto result_op = builder::Tuple({output, mask});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, DropoutGradEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp Mask = *(map_inputs["Mask"].at(0));
  auto prob = PADDLE_GET_CONST(float, op->GetAttr("dropout_prob"));
  auto is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  auto dropout_imple =
      PADDLE_GET_CONST(std::string, op->GetAttr("dropout_implementation"));

  auto prob_1 = 1 - prob;
  auto scalar_type = builder::Type({1}, out_grad.GetType().GetPrimitiveType());

  auto prob1_op = builder::Const(gcu_builder, prob_1, scalar_type);
  auto mask = builder::Convert(
      Mask, {Mask.GetType().GetShape(), out_grad.GetType().GetPrimitiveType()});
  GcuOp in_grad_op;

  if (is_test) {
    if (dropout_imple == "upscale_in_train") {
      in_grad_op = out_grad * mask;
    } else {
      in_grad_op = out_grad * prob1_op;
    }
  } else {
    if (dropout_imple == "upscale_in_train") {
      if (prob == 1.0f) {
        in_grad_op = out_grad * mask;
      } else {
        in_grad_op = out_grad * mask / prob1_op;
      }
    } else {
      in_grad_op = out_grad * mask;
    }
  }
  return std::make_shared<GcuOp>(in_grad_op);
}
EQUIVALENCE_TRANS_FUNC_REG(kDropout, INSENSITIVE, DropoutEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kDropoutGrad,
                           INSENSITIVE,
                           DropoutGradEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
