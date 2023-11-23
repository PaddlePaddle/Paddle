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
const char *const kConcat = "concat";
const char *const kConcatGrad = "concat_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ConcatEquivalenceTrans) {
  auto *op = node->Op();
  auto axis_int = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto axis = static_cast<int64_t>(axis_int);
  std::vector<builder::Op> ops;
  auto input_num = map_inputs["X"].size();
  for (size_t i = 0; i < input_num; ++i) {
    ops.emplace_back(*(map_inputs["X"].at(i)));
  }
  if (axis < 0) axis += ops[0].GetType().GetRank();
  return std::make_shared<GcuOp>(builder::Concatenate(ops, axis));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ConcatGradEquivalenceTrans) {
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto grad_out = *(map_inputs["Out@GRAD"].at(0));
  std::vector<GcuOp> inputs;
  auto input_name_map = op->Inputs();
  std::vector<std::string> input_names = input_name_map["X"];
  for (size_t i = 0; i < input_names.size(); ++i) {
    inputs.push_back(*(map_inputs["X"][i]));
  }
  auto result = builder::ConcatGrad(grad_out, inputs, axis);

  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names = output_name_map["X@GRAD"];
  if (input_names.size() == output_names.size()) {
    std::string output_names_attr(output_names[0]);
    for (size_t i = 1; i < output_names.size(); ++i) {
      output_names_attr += ";" + output_names[i];
    }
    result.SetAttribute(kAttrOpOutVarName,
                        builder::Attribute(output_names_attr.c_str()));
    return std::make_shared<GcuOp>(result);
  } else {
    // maybe return only a part of grad for the inputs
    std::vector<int> output_indices;
    for (const auto &name : output_names) {
      auto prefix = name.substr(0, name.size() - 5);  // XXX@GRAD -> XXX
      auto iter = std::find(input_names.begin(), input_names.end(), prefix);
      if (iter == input_names.end()) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "unvalid output name of : %s", name.c_str()));
      }
      output_indices.push_back(static_cast<int>(iter - input_names.begin()));
    }
    std::vector<builder::Op> outputs;
    for (auto i : output_indices) {
      outputs.emplace_back(builder::GetTupleElement(result, i));
    }
    if (outputs.size() == 1) {
      return std::make_shared<GcuOp>(outputs[0]);
    } else {
      result = builder::Tuple(outputs);
      std::string output_names_attr(output_names[0]);
      for (size_t i = 1; i < output_names.size(); ++i) {
        output_names_attr += ";" + output_names[i];
      }
      result.SetAttribute(kAttrOpOutVarName,
                          builder::Attribute(output_names_attr.c_str()));
      return std::make_shared<GcuOp>(result);
    }
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kConcat, INSENSITIVE, ConcatEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kConcatGrad,
                           INSENSITIVE,
                           ConcatGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
