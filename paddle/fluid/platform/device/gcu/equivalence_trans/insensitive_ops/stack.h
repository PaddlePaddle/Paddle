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
const char* const kStack = "stack";
const char* const kStackGrad = "stack_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, StackEquivalenceTrans) {
  auto X = map_inputs["X"];
  auto axis = PADDLE_GET_CONST(int, node->Op()->GetAttr("axis"));
  std::vector<GcuOp> inputs;
  for (const auto& input : X) {
    inputs.push_back(*input);
  }
  auto out = builder::Stack(inputs, axis);
  return std::make_shared<GcuOp>(out);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, StackGradEquivalenceTrans) {
  auto grad_out = *(map_inputs["Y@GRAD"].at(0));
  auto op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));

  auto out_op = builder::StackGrad(grad_out, axis);
  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names = output_name_map["X@GRAD"];
  std::string output_names_attr(output_names[0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_names[i];
  }
  out_op.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kStack, INSENSITIVE, StackEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kStackGrad, INSENSITIVE, StackGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
