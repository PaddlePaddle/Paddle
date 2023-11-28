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
const char *const kLayerNorm = "layer_norm";
const char *const kLayerNormGrad = "layer_norm_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LayerNormEquivalenceTrans) {
  auto *op = node->Op();
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  int64_t axis = PADDLE_GET_CONST(int, op->GetAttr("begin_norm_axis"));
  builder::Op input_x = *(map_inputs["X"].at(0));
  auto input_shape = input_x.GetType().GetShape();
  int64_t input_rank = input_x.GetType().GetRank();

  std::vector<int64_t> scale_bias_shape;
  int64_t left = 1;
  for (int64_t i = 0; i < axis; ++i) {
    left *= input_shape[i];
  }
  for (int64_t i = axis; i < input_rank; ++i) {
    scale_bias_shape.push_back(input_shape[i]);
  }

  builder::Op scale_op;
  builder::Op bias_op;
  if (map_inputs.count("Scale") != 0 && map_inputs["Scale"].size() != 0) {
    scale_op = *(map_inputs["Scale"].at(0));
    scale_op = builder::Reshape(scale_op, scale_bias_shape);
  } else {
    scale_op = builder::Const(
        gcu_builder,
        1.0f,
        builder::Type(scale_bias_shape, input_x.GetType().GetPrimitiveType()));
  }
  if (map_inputs.count("Bias") != 0 && map_inputs["Bias"].size() != 0) {
    bias_op = *(map_inputs["Bias"].at(0));
    bias_op = builder::Reshape(bias_op, scale_bias_shape);
  } else {
    bias_op = builder::Const(
        gcu_builder,
        0.0f,
        builder::Type(scale_bias_shape, input_x.GetType().GetPrimitiveType()));
  }

  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_shape.push_back({builder::kUnknownRank});
  tuple_shape.push_back({builder::kUnknownRank});
  tuple_shape.push_back({builder::kUnknownRank});
  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::LayerNorm(
      input_x, scale_op, bias_op, axis, epsilon, outputs_type);
  auto out_0 = builder::GetTupleElement(result, 0);
  auto out_1 = builder::GetTupleElement(result, 1);
  auto out_2 = builder::GetTupleElement(result, 2);
  out_1 = builder::Reshape(out_1, {left});
  out_2 = builder::Reshape(out_2, {left});
  result = builder::Tuple({out_0, out_1, out_2});

  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names{"Y", "Mean", "Variance"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               LayerNormGradEquivalenceTrans) {
  auto *op = node->Op();
  builder::Op input_x = *(map_inputs["X"].at(0));
  builder::Op mean = *(map_inputs["Mean"].at(0));
  builder::Op variance = *(map_inputs["Variance"].at(0));
  builder::Op dy = *(map_inputs["Y@GRAD"].at(0));
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  int64_t axis = static_cast<int64_t>(
      PADDLE_GET_CONST(int, op->GetAttr("begin_norm_axis")));
  auto input_shape = input_x.GetType().GetShape();
  int64_t in_rank = input_x.GetType().GetRank();

  std::vector<int64_t> mean_var_shape;
  std::vector<int64_t> scale_bias_shape;
  int64_t right = 1;
  for (int64_t i = 0; i < axis; ++i) {
    mean_var_shape.push_back(input_shape[i]);
  }
  for (int64_t i = axis; i < in_rank; ++i) {
    scale_bias_shape.push_back(input_shape[i]);
    right *= input_shape[i];
  }

  builder::Op scale_op;
  if (map_inputs.count("Scale") != 0 && map_inputs["Scale"].size() != 0) {
    scale_op = *(map_inputs["Scale"].at(0));
    scale_op = builder::Reshape(scale_op, scale_bias_shape);
  } else {
    scale_op = builder::Const(
        gcu_builder,
        1.0f,
        builder::Type(scale_bias_shape, input_x.GetType().GetPrimitiveType()));
  }

  mean = builder::Reshape(mean, mean_var_shape);
  variance = builder::Reshape(variance, mean_var_shape);

  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_dtype.push_back(input_x.GetType().GetPrimitiveType());
  tuple_shape.push_back({builder::kUnknownRank});
  tuple_shape.push_back({builder::kUnknownRank});
  tuple_shape.push_back({builder::kUnknownRank});
  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto layer_out = builder::LayerNormGrad(
      input_x, scale_op, mean, variance, dy, axis, epsilon, outputs_type);
  auto output_name_map = op->Outputs();
  std::string output_names_attr;
  GcuOp x_out = builder::GetTupleElement(layer_out, 0);
  GcuOp s_out = builder::GetTupleElement(layer_out, 1);
  GcuOp b_out = builder::GetTupleElement(layer_out, 2);
  std::vector<builder::Op> layer_list;
  bool x_grad = output_name_map.count("X@GRAD") != 0 &&
                output_name_map["X@GRAD"].size() > 0;
  bool s_grad = output_name_map.count("Scale@GRAD") != 0 &&
                output_name_map["Scale@GRAD"].size() > 0;
  bool b_grad = output_name_map.count("Bias@GRAD") != 0 &&
                output_name_map["Bias@GRAD"].size() > 0;
  if (x_grad) {
    output_names_attr += output_name_map["X@GRAD"].at(0) + ";";
    layer_list.push_back(x_out);
  }
  if (s_grad) {
    output_names_attr += output_name_map["Scale@GRAD"].at(0) + ";";
    s_out = builder::Reshape(s_out, {right});
    layer_list.push_back(s_out);
  }
  if (b_grad) {
    output_names_attr += output_name_map["Bias@GRAD"].at(0) + ";";
    b_out = builder::Reshape(b_out, {right});
    layer_list.push_back(b_out);
  }
  auto output = builder::Tuple(layer_list);
  output.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kLayerNorm, INSENSITIVE, LayerNormEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLayerNormGrad,
                           INSENSITIVE,
                           LayerNormGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
