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
const char *const kReshape = "reshape2";
const char *const kReshapeGrad = "reshape2_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReshapeEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
  if (map_inputs.count("ShapeTensor") != 0) {
    std::vector<int> dims;
    for (size_t i = 0; i < map_inputs["ShapeTensor"].size(); ++i) {
      dims.emplace_back(
          map_inputs["ShapeTensor"].at(0)->GetConstData<int>()[0]);
    }
    shape = dims;
  } else if (map_inputs.count("Shape") != 0) {
    shape = map_inputs["Shape"].at(0)->GetConstData<int>();
  }
  std::vector<int64_t> new_shape;
  for (auto dim : shape) {
    new_shape.emplace_back(static_cast<int64_t>(dim));
  }
  auto raw_shape = input.GetType().GetShape();
  int64_t tmp = 1;
  int negative_dim = -1;
  int dims = static_cast<int>(new_shape.size());
  for (int i = 0; i < dims; ++i) {
    if (new_shape[i] >= 0) {
      if (new_shape[i] == 0) {
        new_shape[i] = raw_shape[i];
      }
      tmp *= new_shape[i];
    } else {
      negative_dim = i;
    }
  }
  if (negative_dim >= 0) {
    auto size = input.GetType().GetSize();
    new_shape[negative_dim] = size / tmp;
  }
  builder::Type output_type(new_shape, input.GetType().GetPrimitiveType());
  auto data = builder::Reshape(input, output_type);

  // Ref: Paddle/paddle/fluid/operators/reshape_op.cc
  auto shape_op = builder::EmptyLike(input);
  auto tuple_result = builder::Tuple({data, shape_op});
  auto output_name_map = op->Outputs();
  std::string output_names_attr =
      output_name_map["Out"][0] + ";" + output_name_map["XShape"][0];
  tuple_result.SetAttribute(kAttrOpOutVarName,
                            builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(tuple_result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ReshapeGradEquivalenceTrans) {
  builder::Op out_grad_op = *(map_inputs["Out@GRAD"].at(0));
  builder::Op in_shape_op = *(map_inputs["XShape"].at(0));

  std::vector<int64_t> src_shape = in_shape_op.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin() + 1, src_shape.end());
  auto out_op = builder::Reshape(out_grad_op, tmp_shape);
  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kReshape, INSENSITIVE, ReshapeEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kReshapeGrad,
                           INSENSITIVE,
                           ReshapeGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
