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
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kRMSProp = "rmsprop";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RMSPropEquivalenceTrans) {
  auto grad_op = *(map_inputs["Grad"].at(0));
  auto param_op = *(map_inputs["Param"].at(0));
  auto moment_op = *(map_inputs["Moment"].at(0));
  auto ms_op = *(map_inputs["MeanSquare"].at(0));
  auto mg_op = *(map_inputs["MeanGrad"].at(0));
  auto lr_op = *(map_inputs["LearningRate"].at(0));

  auto *op = node->Op();
  const bool is_centered = PADDLE_GET_CONST(bool, op->GetAttr("centered"));
  const float epsilon = PADDLE_GET_CONST(float, op->GetAttr("epsilon"));
  const float decay = PADDLE_GET_CONST(float, op->GetAttr("decay"));
  const float momentum = PADDLE_GET_CONST(float, op->GetAttr("momentum"));

  auto rho_op = builder::FullLike(ms_op, decay);
  auto epsilon_op = builder::FullLike(ms_op, epsilon);
  auto decay_op = builder::FullLike(ms_op, decay);
  auto momentum_op = builder::FullLike(ms_op, momentum);
  auto const_1_op = builder::OnesLike(ms_op);

  auto ms_out_op =
      rho_op * ms_op + (const_1_op - rho_op) * builder::Square(grad_op);
  builder::Op mg_out_op, moment_out_op;
  if (is_centered) {
    mg_out_op = rho_op * mg_op + (const_1_op - rho_op) * grad_op;
    moment_out_op =
        momentum_op * moment_op +
        lr_op * grad_op /
            builder::Sqrt(ms_out_op - builder::Square(mg_out_op) + epsilon_op);
  } else {
    mg_out_op = mg_op;
    moment_out_op = momentum_op * moment_op +
                    lr_op * grad_op / builder::Sqrt(ms_out_op + epsilon_op);
  }
  auto param_out_op = param_op - moment_out_op;

  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["MeanGradOut"].at(0) + ";";
  output_names_attr += output_name_map["MeanSquareOut"].at(0) + ";";
  output_names_attr += output_name_map["MomentOut"].at(0) + ";";
  output_names_attr += output_name_map["ParamOut"].at(0);

  auto result_op =
      builder::Tuple({mg_out_op, ms_out_op, moment_out_op, param_out_op});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kRMSProp, INSENSITIVE, RMSPropEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
