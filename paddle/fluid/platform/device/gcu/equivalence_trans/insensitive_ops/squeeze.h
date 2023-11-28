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
const char *const kSqueeze = "squeeze";
const char *const kSqueezeGrad = "squeeze_grad";
const char *const kSqueeze2 = "squeeze2";
const char *const kSqueeze2Grad = "squeeze2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SqueezeEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int32_t> new_axes;
  auto input_shape = input.GetType().GetShape();
  auto input_rank = input.GetType().GetRank();
  for (auto dim : axes) {
    if (dim < 0) dim += input_rank;
    if (input_shape[dim] == 1) new_axes.emplace_back(dim);
  }
  auto axes_op =
      builder::Const(input.GetBuilder(),
                     static_cast<void *>(new_axes.data()),
                     builder::Type({static_cast<int64_t>(new_axes.size())},
                                   builder::PrimitiveType::S32()));
  auto res = builder::Squeeze(input, axes_op);
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SqueezeGradEquivalenceTrans) {
  builder::Op out_grad = *(map_inputs["Out@GRAD"].at(0));
  builder::Op input = *(map_inputs["X"].at(0));

  std::vector<int64_t> new_shapes = input.GetType().GetShape();
  auto res = builder::Reshape(out_grad, new_shapes);
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Squeeze2EquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto axes = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("axes"));
  std::vector<int32_t> new_axes;
  auto input_shape = input.GetType().GetShape();
  auto input_rank = input.GetType().GetRank();
  for (auto dim : axes) {
    if (dim < 0) dim += input_rank;
    if (input_shape[dim] == 1) new_axes.emplace_back(dim);
  }

  auto axes_op =
      builder::Const(input.GetBuilder(),
                     static_cast<void *>(new_axes.data()),
                     builder::Type({static_cast<int64_t>(new_axes.size())},
                                   builder::PrimitiveType::S32()));
  auto res = builder::Squeeze(input, axes_op);

  auto shape_op = builder::EmptyLike(input);
  auto tuple_result = builder::Tuple({res, shape_op});
  auto output_name_map = op->Outputs();
  std::string output_names_attr =
      output_name_map["Out"][0] + ";" + output_name_map["XShape"][0];
  tuple_result.SetAttribute(kAttrOpOutVarName,
                            builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(tuple_result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Squeeze2GradEquivalenceTrans) {
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp in_shape = *(map_inputs["XShape"].at(0));

  std::vector<int64_t> src_shape = in_shape.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin() + 1, src_shape.end());
  builder::Type output_type(tmp_shape, out_grad.GetType().GetPrimitiveType());
  auto out_op = builder::Reshape(out_grad, output_type);

  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kSqueeze, INSENSITIVE, SqueezeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSqueezeGrad,
                           INSENSITIVE,
                           SqueezeGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSqueeze2, INSENSITIVE, Squeeze2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSqueeze2Grad,
                           INSENSITIVE,
                           Squeeze2GradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
