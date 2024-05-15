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
const char *const kWhere = "where";
const char *const kWhereGrad = "where_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, WhereEquivalenceTrans) {
  GcuOp x;
  GcuOp y;
  GcuOp condition = *(map_inputs["Condition"].at(0));
  if (map_inputs.count("X") != 0 && map_inputs["X"].size() != 0) {
    x = *(map_inputs["X"].at(0));
  }

  if (map_inputs.count("Y") != 0 && map_inputs["Y"].size() != 0) {
    y = *(map_inputs["Y"].at(0));
  }
  GcuOp result;
  if (x.IsValid() && y.IsValid()) {
    result = builder::Select(condition, x, y);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("X and Y must Be inputed!"));
  }
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, WhereGradEquivalenceTrans) {
  auto *op = node->Op();
  auto grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto condition = *(map_inputs["Condition"].at(0));
  auto out = builder::SelectGrad(grad_out, condition);

  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0 &&
      output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    std::string output_names_attr =
        output_name_map["X@GRAD"].at(0) + ";" + output_name_map["Y@GRAD"].at(0);
    out.SetAttribute(kAttrOpOutVarName,
                     builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(out);
  } else if (output_name_map.count("X@GRAD") != 0 &&
             output_name_map["X@GRAD"].size() > 0) {
    auto grad_lhs = builder::GetTupleElement(out, 0);
    return std::make_shared<GcuOp>(grad_lhs);
  } else if (output_name_map.count("Y@GRAD") != 0 &&
             output_name_map["Y@GRAD"].size() > 0) {
    auto grad_rhs = builder::GetTupleElement(out, 1);
    return std::make_shared<GcuOp>(grad_rhs);
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("grad_x or grad_y must be output!"));
    return std::make_shared<GcuOp>(builder::Op());
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kWhere, INSENSITIVE, WhereEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kWhereGrad, INSENSITIVE, WhereGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
