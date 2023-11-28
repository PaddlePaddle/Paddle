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

const char *const kFlatten = "flatten";
const char *const kFlattenGrad = "flatten_grad";
const char *const kFlatten2 = "flatten2";
const char *const kFlatten2Grad = "flatten2_grad";
const char *const kFlattenWithXShape = "flatten_with_xshape";
const char *const kFlattenWithXShapeGrad = "flatten_with_xshape_grad";
const char *const kFlattenContiguousRange = "flatten_contiguous_range";
const char *const kFlattenContiguousRangeGrad = "flatten_contiguous_range_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               FlattenContiguousRangeEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp data = *(map_inputs["X"].at(0));
  auto output_name_map = op->Outputs();
  auto start_axis = PADDLE_GET_CONST(int, op->GetAttr("start_axis"));
  auto stop_axis = PADDLE_GET_CONST(int, op->GetAttr("stop_axis"));

  auto result = builder::FlattenV2(data, start_axis, stop_axis);
  auto out_shape_op = builder::EmptyLike(data, builder::PrimitiveType::S64());

  std::vector<std::string> output_names{"Out", "XShape"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  auto res = builder::Tuple({result, out_shape_op});
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));

  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               FlattenContiguousRangeGradEquivalenceTrans) {
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp in_shape = *(map_inputs["XShape"].at(0));

  std::vector<int64_t> src_shape = in_shape.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin() + 1, src_shape.end());
  builder::Type output_type(tmp_shape, out_grad.GetType().GetPrimitiveType());
  auto res = builder::Reshape(out_grad, output_type);

  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FlattenEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp data = *(map_inputs["X"].at(0));
  auto output_name_map = op->Outputs();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto result = builder::Flatten(data, axis);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FlattenGradEquivalenceTrans) {
  GcuOp in = *(map_inputs["X"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto res = builder::Reshape(out_grad, in.GetType().GetShape());
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Flatten2EquivalenceTrans) {
  auto *op = node->Op();
  GcuOp data = *(map_inputs["X"].at(0));
  auto output_name_map = op->Outputs();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));

  auto result = builder::Flatten(data, axis);
  auto out_shape_op = builder::EmptyLike(data, builder::PrimitiveType::S64());

  std::vector<std::string> output_names{"Out", "XShape"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  auto res = builder::Tuple({result, out_shape_op});
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, Flatten2GradEquivalenceTrans) {
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  GcuOp in_shape = *(map_inputs["XShape"].at(0));
  std::vector<int64_t> src_shape = in_shape.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin() + 1, src_shape.end());
  builder::Type output_type(tmp_shape, out_grad.GetType().GetPrimitiveType());
  auto res = builder::Reshape(out_grad, output_type);
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kFlattenContiguousRange,
                           INSENSITIVE,
                           FlattenContiguousRangeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlattenContiguousRangeGrad,
                           INSENSITIVE,
                           FlattenContiguousRangeGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlatten2, INSENSITIVE, Flatten2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlatten2Grad,
                           INSENSITIVE,
                           Flatten2GradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlatten, INSENSITIVE, FlattenEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlattenGrad,
                           INSENSITIVE,
                           FlattenGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlattenWithXShape,
                           INSENSITIVE,
                           FlattenContiguousRangeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlattenWithXShapeGrad,
                           INSENSITIVE,
                           FlattenContiguousRangeGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
