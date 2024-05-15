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
const char *const kTopK = "top_k";
const char *const kTopKV2 = "top_k_v2";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TopKEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp input = *(map_inputs["X"].at(0));
  GcuOp k;
  if (map_inputs.count("K") != 0 && map_inputs["K"].size() != 0) {
    k = *(map_inputs["K"].at(0));
    k = builder::Convert(
        k, {k.GetType().GetShape(), builder::PrimitiveType::S64()});
  } else {
    auto k_value = PADDLE_GET_CONST(int, op->GetAttr("k"));
    auto scalar_type = builder::Type(builder::PrimitiveType::S64());
    k = builder::Const(gcu_builder, k_value, scalar_type);
  }
  int64_t axis = input.GetType().GetRank() - 1;
  GcuOp result =
      builder::TopK(input, k, /*axis=*/axis, /*sorted=*/true, /*largest=*/true);
  std::vector<std::string> output_names{"Out", "Indices"};
  auto output_name_map = op->Outputs();
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TopKV2EquivalenceTrans) {
  auto *op = node->Op();
  GcuOp input = *(map_inputs["X"].at(0));
  GcuOp k;
  if (map_inputs.count("K") != 0 && map_inputs["K"].size() != 0) {
    k = *(map_inputs["K"].at(0));
    k = builder::Convert(
        k, {k.GetType().GetShape(), builder::PrimitiveType::S64()});
  } else {
    auto k_value = PADDLE_GET_CONST(int, op->GetAttr("k"));
    auto scalar_type = builder::Type(builder::PrimitiveType::S64());
    k = builder::Const(gcu_builder, k_value, scalar_type);
  }

  int64_t axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  bool sorted = PADDLE_GET_CONST(bool, op->GetAttr("sorted"));
  bool largest = PADDLE_GET_CONST(bool, op->GetAttr("largest"));
  if (axis < 0) axis += input.GetType().GetRank();
  GcuOp result = builder::TopK(input, k, axis, sorted, largest);
  std::vector<std::string> output_names{"Out", "Indices"};
  auto output_name_map = op->Outputs();
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_name_map[output_names[i]][0];
  }

  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kTopK, INSENSITIVE, TopKEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTopKV2, INSENSITIVE, TopKV2EquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
