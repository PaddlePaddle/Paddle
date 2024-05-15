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
const char *const kAdam = "adam";
const char *const kMergedAdam = "merged_adam";
IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, AdamEquivalenceTrans) {
  builder::Op param_op = *(map_inputs["Param"].at(0));
  builder::Op grad_op = *(map_inputs["Grad"].at(0));
  builder::Op learning_rate_op = *(map_inputs["LearningRate"].at(0));
  builder::Op moment_1_op = *(map_inputs["Moment1"].at(0));
  builder::Op moment_2_op = *(map_inputs["Moment2"].at(0));
  builder::Op beta_1_pow_op = *(map_inputs["Beta1Pow"].at(0));
  builder::Op beta_2_pow_op = *(map_inputs["Beta2Pow"].at(0));

  auto *op = node->Op();

  builder::Op beta_1_op, beta_2_op, epsilon_op;
  if (map_inputs.count("Beta1Tensor") != 0) {
    beta_1_op = *(map_inputs["Beta1Tensor"].at(0));
  } else {
    float beta_1_data = PADDLE_GET_CONST(float, op->GetAttr("beta1"));
    beta_1_op = builder::FullLike(beta_1_pow_op,
                                  beta_1_data,
                                  beta_1_pow_op.GetType().GetPrimitiveType(),
                                  {1});  // directly set shape to 1x
  }
  if (map_inputs.count("Beta2Tensor") != 0) {
    beta_2_op = *(map_inputs["Beta2Tensor"].at(0));
  } else {
    float beta_2_data = PADDLE_GET_CONST(float, op->GetAttr("beta2"));
    beta_2_op = builder::FullLike(beta_2_pow_op,
                                  beta_2_data,
                                  beta_2_pow_op.GetType().GetPrimitiveType(),
                                  {1});  // directly set shape to 1x
  }
  if (map_inputs.count("EpsilonTensor") != 0) {
    epsilon_op = *(map_inputs["EpsilonTensor"].at(0));
  } else {
    float epsilon_data = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
    epsilon_op = builder::FullLike(param_op,
                                   epsilon_data,
                                   param_op.GetType().GetPrimitiveType(),
                                   {1});  // directly set shape to 1x
  }

  bool skip_update = false;
  if (map_inputs.count("SkipUpdate") != 0) {
    skip_update = map_inputs["SkipUpdate"].at(0)->GetConstData<bool>()[0];
  }

  builder::Op param_out_op;
  builder::Op moment_1_out_op, moment_2_out_op;
  builder::Op beta_1_pow_out_op, beta_2_pow_out_op;
  if (skip_update) {
    param_out_op = param_op;
    moment_1_out_op = moment_1_op;
    moment_2_out_op = moment_2_op;
    beta_1_pow_out_op = beta_1_pow_op;
    beta_2_pow_out_op = beta_2_pow_op;
  } else {  // Adam algorithm
    builder::Op const_1_op =
        builder::OnesLike(beta_1_op,
                          beta_1_op.GetType().GetPrimitiveType(),
                          {1});  // directly set shape to 1x

    beta_1_pow_out_op = beta_1_pow_op;
    beta_2_pow_out_op = beta_2_pow_op;

    bool use_global_beta_pow =
        PADDLE_GET_CONST(bool, op->GetAttr("use_global_beta_pow"));
    if (!use_global_beta_pow) {
      beta_1_pow_out_op = beta_1_pow_op * beta_1_op;
      beta_2_pow_out_op = beta_2_pow_op * beta_2_op;
    }

    moment_1_out_op =
        beta_1_op * moment_1_op + (const_1_op - beta_1_op) * grad_op;
    moment_2_out_op =
        beta_2_op * moment_2_op + (const_1_op - beta_2_op) * grad_op * grad_op;

    learning_rate_op =
        learning_rate_op * (builder::Sqrt(const_1_op - beta_2_pow_op) /
                            (const_1_op - beta_1_pow_op));

    param_out_op =
        param_op -
        learning_rate_op *
            (moment_1_out_op / (builder::Sqrt(moment_2_out_op) + epsilon_op));
  }

  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["Beta1PowOut"].at(0) + ";";
  output_names_attr += output_name_map["Beta2PowOut"].at(0) + ";";
  output_names_attr += output_name_map["Moment1Out"].at(0) + ";";
  output_names_attr += output_name_map["Moment2Out"].at(0) + ";";
  output_names_attr += output_name_map["ParamOut"].at(0);

  auto result_op = builder::Tuple({beta_1_pow_out_op,
                                   beta_2_pow_out_op,
                                   moment_1_out_op,
                                   moment_2_out_op,
                                   param_out_op});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MergedAdamEquivalenceTrans) {
  std::vector<GcuOp> param_op;
  std::vector<GcuOp> grad_op;
  std::vector<GcuOp> learning_rate_op;
  std::vector<GcuOp> moment_1_op;
  std::vector<GcuOp> moment_2_op;
  std::vector<GcuOp> beta_1_pow_op;
  std::vector<GcuOp> beta_2_pow_op;
  std::vector<GcuOp> master_param_op;  // Dispensable
  GcuOp beta_1_op, beta_2_op, epsilon_op, const_1_op;

  if (map_inputs["Param"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Param"].size(); ++i)
      param_op.emplace_back(*(map_inputs["Param"][i]));
  }

  if (map_inputs["Grad"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Grad"].size(); ++i)
      grad_op.emplace_back(*(map_inputs["Grad"][i]));
  }

  if (map_inputs["LearningRate"].size() != 0) {
    for (size_t i = 0; i < map_inputs["LearningRate"].size(); ++i)
      learning_rate_op.emplace_back(*(map_inputs["LearningRate"][i]));
  }

  if (map_inputs["Moment1"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Moment1"].size(); ++i)
      moment_1_op.emplace_back(*(map_inputs["Moment1"][i]));
  }

  if (map_inputs["Moment2"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Moment2"].size(); ++i)
      moment_2_op.emplace_back(*(map_inputs["Moment2"][i]));
  }

  if (map_inputs["Beta1Pow"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Beta1Pow"].size(); ++i)
      beta_1_pow_op.emplace_back(*(map_inputs["Beta1Pow"][i]));
  }

  if (map_inputs["Beta2Pow"].size() != 0) {
    for (size_t i = 0; i < map_inputs["Beta2Pow"].size(); ++i)
      beta_2_pow_op.emplace_back(*(map_inputs["Beta2Pow"][i]));
  }

  if (map_inputs["MasterParam"].size() != 0) {
    for (size_t i = 0; i < map_inputs["MasterParam"].size(); ++i)
      master_param_op.emplace_back(*(map_inputs["MasterParam"][i]));
  }

  auto *op = node->Op();
  bool multi_precision = PADDLE_GET_CONST(bool, op->GetAttr("multi_precision"));
  PADDLE_ENFORCE_EQ(
      multi_precision,
      false,
      platform::errors::InvalidArgument("The multi_precision must be equal to "
                                        "false, but got true"));
  PADDLE_ENFORCE_EQ(master_param_op.empty(),
                    true,
                    platform::errors::InvalidArgument(
                        "Gcu backends not support meraged_adam with AMP"));
  const size_t n = param_op.size();
  PADDLE_ENFORCE_EQ(n,
                    grad_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(Grad) must be equal to "
                        "Input(Param), but got the size of Input(Grad) "
                        "is %d, the size of Input(Param) is %d.",
                        grad_op.size(),
                        n));
  PADDLE_ENFORCE_EQ(n,
                    learning_rate_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(LearningRate) must be equal to "
                        "Input(Param), but got the size of Input(LearningRate) "
                        "is %d, the size of Input(Param) is %d.",
                        learning_rate_op.size(),
                        n));
  PADDLE_ENFORCE_EQ(n,
                    moment_1_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(Moment1) must be equal to "
                        "Input(Param), but got the size of Input(Moment1) "
                        "is %d, the size of Input(Param) is %d.",
                        moment_1_op.size(),
                        n));
  PADDLE_ENFORCE_EQ(n,
                    moment_2_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(Moment2) must be equal to "
                        "Input(Param), but got the size of Input(Moment2) "
                        "is %d, the size of Input(Param) is %d.",
                        moment_2_op.size(),
                        n));
  PADDLE_ENFORCE_EQ(n,
                    beta_1_pow_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(Beta1Pow) must be equal to "
                        "Input(Param), but got the size of Input(Beta1Pow) "
                        "is %d, the size of Input(Param) is %d.",
                        beta_1_pow_op.size(),
                        n));
  PADDLE_ENFORCE_EQ(n,
                    beta_2_pow_op.size(),
                    platform::errors::InvalidArgument(
                        "The size of Input(Beta2Pow) must be equal to "
                        "Input(Param), but got the size of Input(Beta2Pow) "
                        "is %d, the size of Input(Param) is %d.",
                        beta_2_pow_op.size(),
                        n));
  std::vector<GcuOp> param_out_op(param_op.size());
  std::vector<GcuOp> moment_1_out_op(moment_1_op.size());
  std::vector<GcuOp> moment_2_out_op(moment_2_op.size());
  std::vector<GcuOp> beta_1_pow_out_op(beta_1_pow_op.size());
  std::vector<GcuOp> beta_2_pow_out_op(beta_2_pow_op.size());
  std::vector<GcuOp> master_param_out_op(master_param_op.size());

  float beta_1_data = PADDLE_GET_CONST(float, op->GetAttr("beta1"));
  float beta_2_data = PADDLE_GET_CONST(float, op->GetAttr("beta2"));
  float epsilon_data = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  bool use_global_beta_pow =
      PADDLE_GET_CONST(bool, op->GetAttr("use_global_beta_pow"));
  for (size_t i = 0; i < n; ++i) {
    auto scalar_type = builder::Type(grad_op[i].GetType().GetPrimitiveType());
    beta_1_op = builder::Const(gcu_builder, beta_1_data, scalar_type);
    beta_2_op = builder::Const(gcu_builder, beta_2_data, scalar_type);
    epsilon_op = builder::Const(gcu_builder, epsilon_data, scalar_type);
    const_1_op = builder::Const(gcu_builder, 1.0, scalar_type);

    beta_1_pow_out_op[i] = beta_1_pow_op[i];
    beta_2_pow_out_op[i] = beta_2_pow_op[i];

    if (!use_global_beta_pow) {
      beta_1_pow_out_op[i] = beta_1_pow_op[i] * beta_1_op;
      beta_2_pow_out_op[i] = beta_2_pow_op[i] * beta_2_op;
    }

    moment_1_out_op[i] =
        beta_1_op * moment_1_op[i] + (const_1_op - beta_1_op) * grad_op[i];
    moment_2_out_op[i] = beta_2_op * moment_2_op[i] +
                         (const_1_op - beta_2_op) * grad_op[i] * grad_op[i];

    learning_rate_op[i] =
        learning_rate_op[i] * (builder::Sqrt(const_1_op - beta_2_pow_op[i]) /
                               (const_1_op - beta_1_pow_op[i]));

    param_out_op[i] =
        param_op[i] - learning_rate_op[i] *
                          (moment_1_out_op[i] /
                           (builder::Sqrt(moment_2_out_op[i]) + epsilon_op));
  }

  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  for (size_t i = 0; i < beta_1_pow_out_op.size(); ++i) {
    output_names_attr += output_name_map["Beta1PowOut"][i] + ";";
  }
  for (size_t i = 0; i < beta_2_pow_out_op.size(); ++i) {
    output_names_attr += output_name_map["Beta2PowOut"][i] + ";";
  }
  for (size_t i = 0; i < moment_1_out_op.size(); ++i) {
    output_names_attr += output_name_map["Moment1Out"][i] + ";";
  }
  for (size_t i = 0; i < moment_2_out_op.size(); ++i) {
    output_names_attr += output_name_map["Moment2Out"][i] + ";";
  }
  for (size_t i = 0; i < param_out_op.size(); ++i) {
    output_names_attr += output_name_map["ParamOut"][i] + ";";
  }
  output_names_attr.resize(output_names_attr.size() - 1);

  std::vector<builder::Op> output_ops;
  for (uint i = 0; i < beta_1_pow_out_op.size(); i++) {
    output_ops.emplace_back(beta_1_pow_out_op[i]);
  }
  for (uint i = 0; i < beta_2_pow_out_op.size(); i++) {
    output_ops.emplace_back(beta_2_pow_out_op[i]);
  }
  for (uint i = 0; i < moment_1_out_op.size(); i++) {
    output_ops.emplace_back(moment_1_out_op[i]);
  }
  for (uint i = 0; i < moment_2_out_op.size(); i++) {
    output_ops.emplace_back(moment_2_out_op[i]);
  }
  for (uint i = 0; i < param_out_op.size(); i++) {
    output_ops.emplace_back(param_out_op[i]);
  }

  auto result_op = builder::Tuple(output_ops);
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kAdam, INSENSITIVE, AdamEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMergedAdam,
                           INSENSITIVE,
                           MergedAdamEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
