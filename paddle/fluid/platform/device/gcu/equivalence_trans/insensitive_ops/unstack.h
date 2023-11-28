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
const char *const kUnstack = "unstack";
const char *const kUnstackGrad = "unstack_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, UnstackEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto num = PADDLE_GET_CONST(int, op->GetAttr("num"));
  auto input_type = input.GetType();
  auto input_rank = static_cast<int>(input_type.GetRank());
  if (axis < 0) {
    axis += input_rank;
  }
  auto dim_at_axis = input_type.GetDimSize(axis);
  if (dim_at_axis > 0 && dim_at_axis != num) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Expect num == x_dim[axis]"));
  }
  std::vector<builder::Op> outputs;
  auto output_slice_shape = input_type.GetShape();
  output_slice_shape.erase(output_slice_shape.begin() + axis);
  for (int i = 0; i < num; ++i) {
    auto slice = builder::SliceInDim(input, i, i + 1, 1, axis);
    outputs.emplace_back(builder::Reshape(slice, output_slice_shape));
  }
  auto result = builder::Tuple(outputs);
  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names = output_name_map["Y"];
  std::string output_names_attr(output_names[0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_names[i];
  }
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, UnstackGradEquivalenceTrans) {
  auto *op = node->Op();
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  std::vector<builder::Op> ops;
  auto input_num = map_inputs["Y@GRAD"].size();
  for (size_t i = 0; i < input_num; ++i) {
    auto slice = *(map_inputs["Y@GRAD"].at(i));
    auto new_shape = slice.GetType().GetShape();
    if (axis < 0) {
      axis += slice.GetType().GetRank() + 1;
    }
    new_shape.insert(new_shape.begin() + axis, 1);
    ops.emplace_back(builder::Reshape(slice, new_shape));
  }
  return std::make_shared<GcuOp>(builder::Concatenate(ops, axis));
}

EQUIVALENCE_TRANS_FUNC_REG(kUnstack, INSENSITIVE, UnstackEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kUnstackGrad,
                           INSENSITIVE,
                           UnstackGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
