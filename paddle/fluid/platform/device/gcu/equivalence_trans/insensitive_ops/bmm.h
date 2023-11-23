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
const char *const kBmm = "bmm";
const char *const kBmmGrad = "bmm_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, BmmEquivalenceTrans) {
  auto lhs = *(map_inputs["X"].at(0));
  auto rhs = *(map_inputs["Y"].at(0));

  std::vector<const char *> precision_config = {};
  builder::DotDimensionNumbers dims_attr({0}, {0}, {2}, {1});
  auto out_op = builder::DotGeneral(lhs, rhs, dims_attr, precision_config);
  out_op.SetAttribute("op_type", builder::Attribute("DotInference"));
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, BmmGradEquivalenceTrans) {
  auto lhs = *(map_inputs["X"].at(0));
  auto rhs = *(map_inputs["Y"].at(0));
  auto grad_out = *(map_inputs["Out@GRAD"].at(0));
  builder::DotDimensionNumbers dims_attr({0}, {0}, {2}, {1});

  GcuOp grad_lhs_op;

  auto *op = node->Op();
  std::vector<const char *> precision_config = {};
  auto output_name_map = op->Outputs();
  if (output_name_map.count("X@GRAD") != 0 &&
      output_name_map["X@GRAD"].size() > 0) {
    auto rhs_transpose = builder::Transpose(rhs, {0, 2, 1});
    grad_lhs_op = builder::DotGeneral(
        grad_out, rhs_transpose, dims_attr, precision_config);
    grad_lhs_op.SetAttribute("op_type", builder::Attribute("DotBPI"));
  }

  GcuOp grad_rhs_op;
  if (output_name_map.count("Y@GRAD") != 0 &&
      output_name_map["Y@GRAD"].size() > 0) {
    auto lhs_transpose = builder::Transpose(lhs, {0, 2, 1});
    grad_rhs_op = builder::DotGeneral(
        lhs_transpose, grad_out, dims_attr, precision_config);
    grad_rhs_op.SetAttribute("op_type", builder::Attribute("DotBPK"));
  }

  if (grad_lhs_op.IsValid() && grad_rhs_op.IsValid()) {
    std::string output_names_attr =
        output_name_map["X@GRAD"][0] + ";" + output_name_map["Y@GRAD"][0];
    auto result = builder::Tuple({grad_lhs_op, grad_rhs_op});
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else if (grad_lhs_op.IsValid()) {
    return std::make_shared<GcuOp>(grad_lhs_op);
  } else {
    return std::make_shared<GcuOp>(grad_rhs_op);
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kBmm, INSENSITIVE, BmmEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kBmmGrad, INSENSITIVE, BmmGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
