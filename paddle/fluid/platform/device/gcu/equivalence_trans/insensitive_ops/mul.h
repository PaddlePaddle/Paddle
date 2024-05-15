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

const char *const kFlattenMul = "mul";
const char *const kFlattenMulGrad = "mul_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FlattenMulEquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  auto input_shape = input.GetType().GetShape();

  int64_t in = 0;
  int64_t ic = 0;
  int64_t ih = 0;
  int64_t iw = 0;
  in = input_shape[0];
  ic = input_shape[1];
  ih = input_shape[2];
  iw = input_shape[3];

  int64_t flattend_size = ic * ih * iw;

  std::vector<int64_t> new_shape;

  new_shape.emplace_back(static_cast<int64_t>(in));
  new_shape.emplace_back(static_cast<int64_t>(flattend_size));

  builder::Type output_type(new_shape, input.GetType().GetPrimitiveType());
  auto data = builder::Reshape(input, output_type);

  builder::DotDimensionNumbers dims_attr({}, {}, {1}, {0});
  std::vector<const char *> precision_config = {};
  auto result = builder::DotGeneral(
      data, *(map_inputs["Y"].at(0)), dims_attr, precision_config);
  result.SetAttribute("op_type", builder::Attribute("DotInference"));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               FlattenMulGradEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp X = *(map_inputs["X"].at(0));
  GcuOp Y = *(map_inputs["Y"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));

  auto x_shape = X.GetType().GetShape();
  auto y_shape = Y.GetType().GetShape();
  std::vector<std::vector<int64_t>> tuple_shape = {{x_shape}, {y_shape}};

  int64_t in = 0;
  int64_t ic = 0;
  int64_t ih = 0;
  int64_t iw = 0;
  in = x_shape[0];
  ic = x_shape[1];
  ih = x_shape[2];
  iw = x_shape[3];

  int64_t flattend_size = ic * ih * iw;
  std::vector<int64_t> new_shape;

  new_shape.emplace_back(static_cast<int64_t>(in));
  new_shape.emplace_back(static_cast<int64_t>(flattend_size));

  builder::Type output_type1(new_shape, X.GetType().GetPrimitiveType());
  auto X_Reshape = builder::Reshape(X, output_type1);

  std::vector<int64_t> data_trans = {1, 0};
  auto XT = builder::Transpose(X_Reshape, data_trans);
  auto YT = builder::Transpose(Y, data_trans);

  builder::DotDimensionNumbers dims_attr({}, {}, {1}, {0});
  std::vector<const char *> precision_config = {};
  auto DX = builder::DotGeneral(out_grad, YT, dims_attr, precision_config);
  auto DY = builder::DotGeneral(XT, out_grad, dims_attr, precision_config);

  DX.SetAttribute("op_type", builder::Attribute("DotBPI"));
  DY.SetAttribute("op_type", builder::Attribute("DotBPK"));

  builder::Type output_type2(tuple_shape[0], X.GetType().GetPrimitiveType());
  auto DX_Reshape = builder::Reshape(DX, output_type2);

  std::vector<GcuOp> outputs;
  outputs.push_back(DX_Reshape);
  outputs.push_back(DY);

  std::vector<builder::PrimitiveType> tuple_dtype;
  tuple_dtype.push_back(DX_Reshape.GetType().GetPrimitiveType());
  tuple_dtype.push_back(DY.GetType().GetPrimitiveType());

  GcuType outputs_type(tuple_shape, tuple_dtype);
  std::vector<std::string> output_names{"X@GRAD", "Y@GRAD"};
  auto output_name_map = op->Outputs();
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  auto res = builder::Tuple(outputs, outputs_type);
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));

  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kFlattenMul,
                           INSENSITIVE,
                           FlattenMulEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kFlattenMulGrad,
                           INSENSITIVE,
                           FlattenMulGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
