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
#include <set>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kBatchNorm = "batch_norm";
const char *const kBatchNormGrad = "batch_norm_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, BatchNormEquivalenceTrans) {
  auto *op = node->Op();
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto momentum = PADDLE_GET_CONST(float, op->GetAttr("momentum"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  auto is_test = PADDLE_GET_CONST(bool, op->GetAttr("is_test"));
  auto trainable_stats =
      PADDLE_GET_CONST(bool, op->GetAttr("trainable_statistics"));
  bool test_mode = is_test && (!trainable_stats);
  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp scale = *(map_inputs["Scale"].at(0));
  GcuOp bias = *(map_inputs["Bias"].at(0));
  GcuOp mean = *(map_inputs["Mean"].at(0));
  GcuOp variance = *(map_inputs["Variance"].at(0));
  std::vector<int64_t> x_perm = {0, 2, 3, 1};
  std::vector<int64_t> y_perm = {0, 3, 1, 2};

  // put ouputs in order of: Y, MeanOut, VarianceOut, SavedMean, SavedVariance,
  // ReserveSpace
  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names{
      "Y", "MeanOut", "VarianceOut", "SavedMean", "SavedVariance"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  if (!test_mode) {
    int64_t N = 1;
    int feature_index = 3;
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, x_perm);
    }

    auto ptype = input_x.GetType().GetPrimitiveType();
    std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
    auto input_shape = input_x.GetType().GetShape();
    int64_t channel_num = input_shape[feature_index];
    std::vector<std::vector<int64_t>> tuple_shape{
        input_shape, {channel_num}, {channel_num}};

    N = input_shape[0] * input_shape[1] * input_shape[2];
    double ratio = static_cast<double>(N) / (N - 1);
    auto fix_factor = TransformUtil::GetConst(gcu_builder, ptype, ratio);

    GcuType batch_normal_outputs_type(tuple_shape, tuple_dtype);
    auto tuple = builder::BatchNormTraining(input_x,
                                            scale,
                                            bias,
                                            epsilon,
                                            feature_index,
                                            batch_normal_outputs_type);
    auto current_mean = builder::GetTupleElement(tuple, 1);
    auto current_variance = builder::GetTupleElement(tuple, 2);

    std::vector<float> v_momentum(channel_num, momentum);
    auto momentum_op = builder::Const(gcu_builder,
                                      static_cast<void *>(v_momentum.data()),
                                      GcuType({channel_num}, ptype));
    std::vector<float> v_momentum_sum(channel_num, 1 - momentum);
    auto momentum_sub_op =
        builder::Const(gcu_builder,
                       static_cast<void *>(v_momentum_sum.data()),
                       GcuType({channel_num}, ptype));
    auto running_mean = mean * momentum_op + current_mean * momentum_sub_op;

    auto running_variance = variance * momentum_op +
                            current_variance * fix_factor * momentum_sub_op;

    auto output_y = builder::GetTupleElement(tuple, 0);
    if (data_layout == "NCHW") {
      output_y = builder::Transpose(output_y, y_perm);
    }

    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance};

    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    tuple_shape.clear();
    tuple_dtype.clear();
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    GcuType outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else {
    if (data_layout == "NCHW") {
      input_x = builder::Transpose(input_x, x_perm);
    }
    int feature_index = 3;
    auto input_shape = input_x.GetType().GetShape();
    auto ptype = input_x.GetType().GetPrimitiveType();
    GcuType batch_normal_outputs_type(input_shape, ptype);
    auto output_y = builder::BatchNormInference(input_x,
                                                scale,
                                                bias,
                                                mean,
                                                variance,
                                                epsilon,
                                                feature_index,
                                                batch_normal_outputs_type);
    if (data_layout == "NCHW") {
      output_y = builder::Transpose(output_y, y_perm);
    }
    auto running_mean = builder::Reshape(mean, mean.GetType());
    auto running_variance = builder::Reshape(variance, variance.GetType());
    builder::Op current_mean;
    if (output_name_map.count("SavedMean") != 0) {
      std::vector<int64_t> saved_mean_shape = {};
      auto saved_mean_op_name = output_name_map["SavedMean"][0];
      for (auto out : node->outputs) {
        if (out->Name() == saved_mean_op_name) {
          saved_mean_shape = out->Var()->GetShape();
          break;
        }
      }
      current_mean = builder::ZerosLike(
          mean, mean.GetType().GetPrimitiveType(), saved_mean_shape);
    }

    builder::Op current_variance;
    if (output_name_map.count("SavedVariance") != 0) {
      std::vector<int64_t> saved_variance_shape = {};
      auto saved_variance_op_name = output_name_map["SavedVariance"][0];
      for (auto out : node->outputs) {
        if (out->Name() == saved_variance_op_name) {
          saved_variance_shape = out->Var()->GetShape();
          break;
        }
      }
      current_variance =
          builder::ZerosLike(variance,
                             variance.GetType().GetPrimitiveType(),
                             saved_variance_shape);
    }
    std::vector<GcuOp> outputs{output_y,
                               running_mean,
                               running_variance,
                               current_mean,
                               current_variance};

    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_name_map[output_names[i]][0];
    }
    std::vector<builder::PrimitiveType> tuple_dtype;
    std::vector<std::vector<int64_t>> tuple_shape;
    for (uint i = 0; i < outputs.size(); i++) {
      tuple_shape.push_back(outputs[i].GetType().GetShape());
      tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
    }
    GcuType outputs_type(tuple_shape, tuple_dtype);
    auto result = builder::Tuple(outputs, outputs_type);
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               BatchNormGradEquivalenceTrans) {
  auto *op = node->Op();
  auto epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  auto data_layout = PADDLE_GET_CONST(std::string, op->GetAttr("data_layout"));
  auto output_name_map = op->Outputs();

  GcuOp input_x = *(map_inputs["X"].at(0));
  GcuOp scale = *(map_inputs["Scale"].at(0));
  GcuOp mean = *(map_inputs["SavedMean"].at(0));
  GcuOp variance = *(map_inputs["SavedVariance"].at(0));
  GcuOp y_grad = *(map_inputs["Y@GRAD"].at(0));
  int feature_index = 3;
  std::vector<int64_t> x_perm = {0, 2, 3, 1};
  std::vector<int64_t> y_perm = {0, 3, 1, 2};
  auto out_type = input_x.GetType();
  auto x_shape = input_x.GetType().GetShape();
  if (data_layout == "NCHW") {
    input_x = builder::Transpose(input_x, x_perm);
    y_grad = builder::Transpose(y_grad, x_perm);
  }
  auto x_trans_shape = input_x.GetType().GetShape();
  auto y_grad_shape = y_grad.GetType().GetShape();
  if (x_trans_shape.size() != 4) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Input tensor must be 4-dimensional, current is: %s",
        std::to_string(x_trans_shape.size())));
  }
  std::set<std::string> formats{"NHWC", "NCHW"};
  if (formats.count(data_layout) == 0) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Invalid format for BatchNormGrad: %s", data_layout));
  }
  if (x_trans_shape[feature_index] != scale.GetType().GetShape()[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Scale size must be equal to input channel"));
  }
  if (x_trans_shape[feature_index] != variance.GetType().GetShape()[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Variance size must be equal to input channel"));
  }
  if (x_trans_shape[feature_index] != mean.GetType().GetShape()[0]) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Mean size must be equal to input channel"));
  }

  auto ptype = input_x.GetType().GetPrimitiveType();
  std::vector<builder::PrimitiveType> tuple_dtype(3, ptype);
  std::vector<std::vector<int64_t>> tuple_shape{x_trans_shape,
                                                {x_trans_shape[feature_index]},
                                                {x_trans_shape[feature_index]}};
  GcuType batch_normal_outputs_type(tuple_shape, tuple_dtype);

  auto gradop = builder::BatchNormGrad(input_x,
                                       scale,
                                       mean,
                                       variance,
                                       y_grad,
                                       epsilon,
                                       feature_index,
                                       batch_normal_outputs_type);

  std::vector<GcuOp> output;
  std::vector<std::vector<int64_t>> output_shape;
  std::vector<builder::PrimitiveType> output_dtype;
  std::vector<std::string> output_name;
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    auto tout = builder::GetTupleElement(gradop, 0);
    if (data_layout == "NCHW") tout = builder::Transpose(tout, y_perm);
    output.push_back(tout);
    output_shape.push_back(x_shape);
    output_dtype.push_back(ptype);
    output_name.push_back("X@GRAD");
  }
  if (output_name_map.count("Scale@GRAD") != 0 &&
      output_name_map["Scale@GRAD"].size() > 0) {
    auto gte1 = builder::GetTupleElement(gradop, 1);
    output.push_back(gte1);
    output_shape.push_back({x_trans_shape[feature_index]});
    output_dtype.push_back(ptype);
    output_name.push_back("Scale@GRAD");
  }
  if (output_name_map.count("Bias@GRAD") != 0 &&
      output_name_map["Bias@GRAD"].size() > 0) {
    auto gte2 = builder::GetTupleElement(gradop, 2);
    output.push_back(gte2);
    output_shape.push_back({x_trans_shape[feature_index]});
    output_dtype.push_back(ptype);
    output_name.push_back("Bias@GRAD");
  }

  GcuType output_type(output_shape, output_dtype);

  std::string output_names_attr{output_name_map[output_name[0]][0]};
  for (size_t i = 1; i < output_name.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_name[i]][0];
  }
  auto res = builder::Tuple(output, output_type);
  res.SetAttribute(kAttrOpOutVarName,
                   builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kBatchNorm, INSENSITIVE, BatchNormEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kBatchNormGrad,
                           INSENSITIVE,
                           BatchNormGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
