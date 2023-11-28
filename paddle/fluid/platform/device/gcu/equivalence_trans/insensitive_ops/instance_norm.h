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

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace platform {
namespace gcu {
const char* const kInstanceNorm = "instance_norm";
const char* const kInstanceNormGrad = "instance_norm_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, InstanceNormEquivalenceTrans) {
  auto* op = node->Op();
  GcuOp input_x = *(map_inputs["X"].at(0));

  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  // auto data_layout = PADDLE_GET_CONST(std::string,
  // op->GetAttr("data_format"));
  auto input_shape_original = input_x.GetType().GetShape();
  auto input_shape = input_shape_original;
  auto rank = input_shape.size();
  if (rank <= 2 || rank > 5) {
    PADDLE_THROW(
        platform::errors::InvalidArgument(" expected 3D or 4D or 5D input."));
  }

  if (rank == 3) {
    input_x = builder::Reshape(
        input_x,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2],
                       1},
                      input_x.GetType().GetPrimitiveType()));

  } else if (rank == 4) {
    input_x =
        builder::Reshape(input_x,
                         builder::Type({1,
                                        input_shape[0] * input_shape[1],
                                        input_shape[2],
                                        input_shape[3]},
                                       input_x.GetType().GetPrimitiveType()));
  } else if (rank == 5) {
    input_x =
        builder::Reshape(input_x,
                         builder::Type({1,
                                        input_shape[0] * input_shape[1],
                                        input_shape[2] * input_shape[3],
                                        input_shape[4]},
                                       input_x.GetType().GetPrimitiveType()));
  }

  input_x = builder::Transpose(input_x, {0, 2, 3, 1});

  input_shape = input_x.GetType().GetShape();

  const int64_t N = input_shape_original[0];
  const int64_t NxC = input_shape[3];

  builder::Op scale;
  if (map_inputs.count("Scale") != 0) {
    scale = *(map_inputs["Scale"].at(0));
    std::vector<builder::Op> scales(N, scale);
    scale = builder::Concatenate(scales, 0);
  } else {
    scale =
        builder::OnesLike(input_x, input_x.GetType().GetPrimitiveType(), {NxC});
  }
  builder::Op bias;
  if (map_inputs.count("Bias") != 0) {
    bias = *(map_inputs["Bias"].at(0));
    std::vector<builder::Op> biases(N, bias);
    bias = builder::Concatenate(biases, 0);
  } else {
    bias = builder::ZerosLike(
        input_x, input_x.GetType().GetPrimitiveType(), {NxC});
  }

  auto ptype = input_x.GetType().GetPrimitiveType();
  std::vector<builder::PrimitiveType> tuple_dtype_x(3, ptype);
  int64_t feature_index = 3;
  int64_t channel_num = input_shape[feature_index];
  std::vector<std::vector<int64_t>> tuple_shape_x{
      input_shape, {channel_num}, {channel_num}};
  GcuType batch_normal_outputs_type(tuple_shape_x, tuple_dtype_x);
  auto outs = builder::BatchNormTraining(
      input_x, scale, bias, epsilon, feature_index, batch_normal_outputs_type);
  auto output_y = builder::GetTupleElement(outs, 0);

  output_y = builder::Transpose(output_y, {0, 3, 1, 2});
  output_y =
      builder::Reshape(output_y,
                       builder::Type(input_shape_original,
                                     output_y.GetType().GetPrimitiveType()));

  auto batch_mean = builder::GetTupleElement(outs, 1);
  auto batch_var = builder::GetTupleElement(outs, 2);
  auto shape = batch_var.GetType().GetShape();

  auto op_epsilon = builder::FullLike(
      input_x, epsilon, input_x.GetType().GetPrimitiveType(), {NxC});
  auto var_eps = builder::Rsqrt(batch_var + op_epsilon);

  std::vector<builder::Op> outputs{output_y, batch_mean, var_eps};
  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names{"Y", "SavedMean", "SavedVariance"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }
  builder::Type outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs, outputs_type);
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               InstanceNormGradEquivalenceTrans) {
  auto* op = node->Op();
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  // auto data_layout = PADDLE_GET_CONST(std::string,
  // op->GetAttr("data_format"));
  auto output_name_map = op->Outputs();

  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp mean = *(map_inputs["SavedMean"].at(0));
  GcuOp variance = *(map_inputs["SavedVariance"].at(0));
  GcuOp y_grad = *(map_inputs["Y@GRAD"].at(0));

  auto mean_shape = mean.GetType().GetShape();
  auto variance_shape = variance.GetType().GetShape();

  auto input_shape_original = input_x.GetType().GetShape();
  auto input_shape = input_shape_original;
  auto rank = input_shape.size();

  if (rank <= 2 || rank > 5) {
    PADDLE_THROW(
        platform::errors::InvalidArgument(" expected 3D or 4D or 5D input."));
  }

  if (rank == 3) {
    input_x = builder::Reshape(
        input_x,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2],
                       1},
                      input_x.GetType().GetPrimitiveType()));
    y_grad = builder::Reshape(
        y_grad,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2],
                       1},
                      y_grad.GetType().GetPrimitiveType()));

  } else if (rank == 4) {
    input_x = builder::Reshape(
        input_x,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2],
                       input_shape_original[3]},
                      input_x.GetType().GetPrimitiveType()));

    y_grad = builder::Reshape(
        y_grad,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2],
                       input_shape_original[3]},
                      y_grad.GetType().GetPrimitiveType()));

  } else if (rank == 5) {
    input_x = builder::Reshape(
        input_x,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2] * input_shape_original[3],
                       input_shape_original[4]},
                      input_x.GetType().GetPrimitiveType()));

    y_grad = builder::Reshape(
        y_grad,
        builder::Type({1,
                       input_shape_original[0] * input_shape_original[1],
                       input_shape_original[2] * input_shape_original[3],
                       input_shape_original[4]},
                      y_grad.GetType().GetPrimitiveType()));
  }
  input_x = builder::Transpose(input_x, {0, 2, 3, 1});
  y_grad = builder::Transpose(y_grad, {0, 2, 3, 1});
  input_shape = input_x.GetType().GetShape();

  if (input_shape.size() != 4) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Input tensor must be 4-dimensional, current is: %s",
        std::to_string(input_shape.size())));
  }

  const int64_t N = input_shape_original[0];
  const int64_t C = input_shape_original[1];
  const int64_t NxC = input_shape[3];

  PADDLE_ENFORCE_EQ(
      N * C, NxC, platform::errors::InvalidArgument("interanl error"));

  builder::Op scale;
  if (map_inputs.count("Scale") != 0) {
    scale = *(map_inputs["Scale"].at(0));
    std::vector<builder::Op> scales(input_shape_original[0], scale);
    scale = builder::Concatenate(scales, 0);
  } else {
    scale =
        builder::OnesLike(input_x, input_x.GetType().GetPrimitiveType(), {NxC});
  }

  auto scale_shape = scale.GetType().GetShape();

  if (NxC != scale_shape[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Scale size must be equal to input channel"));
  }
  if (NxC != variance_shape[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Variance size must be equal to input channel"));
  }
  if (NxC != mean_shape[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Mean size must be equal to input channel"));
  }

  auto input_ty = builder::Type({1, input_shape[1], input_shape[2], NxC},
                                input_x.GetType().GetPrimitiveType());

  auto ptype = input_x.GetType().GetPrimitiveType();
  std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
  std::vector<std::vector<int64_t>> tuple_shape{input_shape, {NxC}, {NxC}};
  GcuType batch_normal_outputs_type(tuple_shape, tuple_dtype);

  auto op_epsilon = builder::FullLike(
      input_x, epsilon, input_x.GetType().GetPrimitiveType(), {NxC});
  auto op_one =
      builder::OnesLike(input_x, input_x.GetType().GetPrimitiveType(), {NxC});
  variance = builder::Div(op_one, variance);
  variance = builder::Mul(variance, variance);
  variance = builder::Sub(variance, op_epsilon);

  auto gradop = builder::BatchNormGrad(input_x,
                                       scale,
                                       mean,
                                       variance,
                                       y_grad,
                                       epsilon,
                                       3,
                                       batch_normal_outputs_type);

  auto gte0 = builder::GetTupleElement(gradop, 0);
  auto gte1 = builder::GetTupleElement(gradop, 1);
  auto gte2 = builder::GetTupleElement(gradop, 2);

  gte0 = builder::Transpose(gte0, {0, 3, 1, 2});
  gte0 = builder::Reshape(
      gte0,
      builder::Type(input_shape_original, gte0.GetType().GetPrimitiveType()));

  gte1 = builder::Reshape(
      gte1, builder::Type({N, C}, gte1.GetType().GetPrimitiveType()));

  gte2 = builder::Reshape(
      gte2, builder::Type({N, C}, gte2.GetType().GetPrimitiveType()));

  gte1 = builder::ReduceSum(gte1, false, {0});
  gte2 = builder::ReduceSum(gte2, false, {0});

  auto gte0_shape = gte0.GetType().GetShape();
  auto gte1_shape = gte1.GetType().GetShape();
  auto gte2_shape = gte2.GetType().GetShape();

  std::vector<GcuOp> outputs;
  outputs.push_back(gte0);
  outputs.push_back(gte1);
  outputs.push_back(gte2);

  tuple_shape = {input_shape_original,
                 {gte1.GetType().GetShape()[0]},
                 {gte2.GetType().GetShape()[0]}};
  GcuType outputs_type(tuple_shape, tuple_dtype);

  std::vector<std::string> output_names{"X@GRAD", "Scale@GRAD", "Bias@GRAD"};
  std::string output_names_attr{output_name_map["X@GRAD"][0]};
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  auto res = builder::Tuple(outputs, outputs_type);
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kInstanceNorm,
                           INSENSITIVE,
                           InstanceNormEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kInstanceNormGrad,
                           INSENSITIVE,
                           InstanceNormGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
