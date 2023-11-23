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
const char *const kMomentum = "momentum";
const char *const kMergedMomentum = "merged_momentum";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MomentumEquivalenceTrans) {
  builder::Op param_op = *(map_inputs["Param"].at(0));
  builder::Op grad_op = *(map_inputs["Grad"].at(0));
  builder::Op velocity_op = *(map_inputs["Velocity"].at(0));
  builder::Op learning_rate_op = *(map_inputs["LearningRate"].at(0));
  auto *op = node->Op();
  float mu = PADDLE_GET_CONST(float, op->GetAttr("mu"));
  bool use_nesterov = PADDLE_GET_CONST(bool, op->GetAttr("use_nesterov"));
  auto regularization_method =
      PADDLE_GET_CONST(std::string, op->GetAttr("regularization_method"));
  float regularization_coeff =
      PADDLE_GET_CONST(float, op->GetAttr("regularization_coeff"));

  if (regularization_method == "l2_decay") {
    auto regular_coeff_op = builder::FullLike(param_op, regularization_coeff);
    auto regular_param = param_op * regular_coeff_op;
    grad_op = grad_op + regular_param;
  } else if (regularization_method.size() != 0) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("GCU Not support regularization"));
  }
  builder::Op param_out_op;
  builder::Op velocity_out;
  auto mu_op = builder::FullLike(velocity_op, mu);

  velocity_out = mu_op * velocity_op + grad_op;

  if (use_nesterov) {
    param_out_op =
        param_op - (grad_op + mu_op * velocity_out) * learning_rate_op;
  } else {
    param_out_op = param_op - learning_rate_op * velocity_out;
  }

  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["VelocityOut"].at(0) + ";";
  output_names_attr += output_name_map["ParamOut"].at(0);

  auto result_op = builder::Tuple({velocity_out, param_out_op});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               MergedMomentumEquivalenceTrans) {
  builder::Op learning_rate_op = *(map_inputs["LearningRate"].at(0));
  std::vector<builder::Op> param_op;
  std::vector<builder::Op> grad_op;
  std::vector<builder::Op> velocity_op;
  if (map_inputs["Param"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Param"].size(); ++i) {
      param_op.emplace_back(*(map_inputs["Param"][i]));
    }
  }
  if (map_inputs["Grad"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Grad"].size(); ++i) {
      grad_op.emplace_back(*(map_inputs["Grad"][i]));
    }
  }
  if (map_inputs["Velocity"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Velocity"].size(); ++i) {
      velocity_op.emplace_back(*(map_inputs["Velocity"][i]));
    }
  }
  auto *op = node->Op();
  float mu = PADDLE_GET_CONST(float, op->GetAttr("mu"));
  bool use_nesterov = PADDLE_GET_CONST(bool, op->GetAttr("use_nesterov"));
  auto regularization_method = PADDLE_GET_CONST(
      std::vector<std::string>, op->GetAttr("regularization_method"));
  auto regularization_coeff =
      PADDLE_GET_CONST(std::vector<float>, op->GetAttr("regularization_coeff"));

  auto lr_shape = learning_rate_op.GetType().GetShape();
  if (param_op.size() != grad_op.size()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The size of Input(Grad) must be equal to Input(Param), but got "
        "the size of Input(Grad) is %d, the size of Input(Param) is %d.",
        grad_op.size(),
        param_op.size()));
  }
  if (param_op.size() != velocity_op.size()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The size of Input(Velocity) must be equal to "
        "Input(Param), but got the size of Input(Velocity) "
        "is %d, the size of Input(Param) is %d.",
        velocity_op.size(),
        param_op.size()));
  }
  if (lr_shape.size() != 1 && lr_shape.size() != 0) {
    if (lr_shape.size() != param_op.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "If the size of Input(LearningRate) is not 1, the size of "
          "Input(LearningRate) must be "
          "equal to Input(Param), but got the size of Input(LearningRate) "
          "is %d, the size of Input(Param) is %d.",
          lr_shape.size(),
          param_op.size()));
    }
  }
  if (regularization_method.size() != 0) {
    if (regularization_method.size() != param_op.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The size of Attr(regularization_method) must be equal "
          "to Input(Param), but got the size of "
          "Attr(regularization_method) is %d, the size of Input(Param) is "
          "%d.",
          regularization_method.size(),
          param_op.size()));
    }
    if (regularization_coeff.size() != param_op.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The size of Attr(regularization_coeff) must be equal "
          "to Input(Param), but got the size of Attr(regularization_coeff) "
          "is %d, the size of Input(Param) is %d.",
          regularization_coeff.size(),
          param_op.size()));
    }
  }

  std::vector<builder::Op> param_out_op;
  std::vector<builder::Op> velocity_out;
  auto mu_op = builder::Const(gcu_builder,
                              static_cast<void *>(&mu),
                              builder::Type({}, builder::PrimitiveType::F32()));
  int n_dims = static_cast<int>(param_op.size());
  for (int i = 0; i < n_dims; ++i) {
    std::string regtype = regularization_method.size() > 0 &&
                                  regularization_method[i] == "l2_decay"
                              ? "l2_decay"
                              : "";
    builder::Op regularized_grad;
    if (regtype == "l2_decay") {
      builder::Op regularization_coeff_value;
      if (regularization_coeff.size() != 0) {
        regularization_coeff_value =
            builder::FullLike(param_op[i], regularization_coeff[i]);
      } else {
        regularization_coeff_value = builder::ZerosLike(param_op[i]);
      }
      auto regular_param = param_op[i] * regularization_coeff_value;
      regularized_grad = grad_op[i] + regular_param;
    } else {
      regularized_grad = grad_op[i];
    }
    builder::Op param_out_op_tmp;
    auto velocity_out_tmp = mu_op * velocity_op[i] + regularized_grad;
    if (use_nesterov) {
      param_out_op_tmp =
          param_op[i] -
          (regularized_grad + mu_op * velocity_out_tmp) * learning_rate_op;
    } else {
      param_out_op_tmp = param_op[i] - learning_rate_op * velocity_out_tmp;
    }
    velocity_out.emplace_back(velocity_out_tmp);
    param_out_op.emplace_back(param_out_op_tmp);
  }
  if (param_op.size() != param_out_op.size()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The size of Output(ParamOut) must be equal to "
        "Input(Param), but got the size of Output(ParamOut) "
        "is %d, the size of Input(Param) is %d.",
        param_out_op.size(),
        param_op.size()));
  }
  for (size_t i = 0; i < param_op.size(); ++i) {
    auto param_shape = param_op[i].GetType().GetShape();
    auto param_out_shape = param_out_op[i].GetType().GetShape();
    if (param_shape.size() != param_out_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The size of Input(Param) and Output(ParamOut) must be the same "
          "Tensors."));
    }
  }
  if (param_op.size() != velocity_out.size()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "The size of Output(VelocityOut) must be "
        "equal to Input(Param), but got the size of Output(VelocityOut) is "
        "%d, the size of Input(Param) is %d.",
        velocity_out.size(),
        param_op.size()));
  }
  for (size_t i = 0; i < param_op.size(); ++i) {
    auto velocity_op_shape = velocity_op[i].GetType().GetShape();
    auto velocity_out_shape = velocity_out[i].GetType().GetShape();
    if (velocity_op_shape.size() != velocity_out_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Input(Velocity) and Output(VelocityOut) must be the same "
          "Tensors."));
    }
  }
  auto output_name_map = op->Outputs();
  std::string output_names_attr = output_name_map["VelocityOut"][0];
  for (size_t i = 1; i < velocity_out.size(); ++i) {
    output_names_attr += ";" + output_name_map["VelocityOut"][i];
  }
  for (size_t i = 0; i < param_out_op.size(); ++i) {
    output_names_attr += ";" + output_name_map["ParamOut"][i];
  }
  std::vector<builder::PrimitiveType> out_dtype;
  std::vector<std::vector<int64_t>> out_shape;
  std::vector<builder::Op> output_op;
  for (uint i = 0; i < velocity_out.size(); i++) {
    out_shape.emplace_back(velocity_out[i].GetType().GetShape());
    out_dtype.emplace_back(velocity_out[i].GetType().GetPrimitiveType());
    output_op.emplace_back(velocity_out[i]);
  }
  for (uint i = 0; i < param_out_op.size(); i++) {
    out_shape.emplace_back(param_out_op[i].GetType().GetShape());
    out_dtype.emplace_back(param_out_op[i].GetType().GetPrimitiveType());
    output_op.emplace_back(param_out_op[i]);
  }
  builder::Type output_type(out_shape, out_dtype);
  auto result_op = builder::Tuple(output_op, output_type);
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kMomentum, INSENSITIVE, MomentumEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMergedMomentum,
                           INSENSITIVE,
                           MergedMomentumEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
