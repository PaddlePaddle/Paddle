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
const char *const kSplit = "split";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SplitEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  auto num = PADDLE_GET_CONST(int, op->GetAttr("num"));
  auto sections = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("sections"));

  if (map_inputs.count("AxisTensor") != 0) {
    axis = map_inputs["AxisTensor"].at(0)->GetConstData<int>()[0];
  }
  if (map_inputs.count("SectionsTensorList") != 0) {
    sections.clear();
    for (size_t i = 0; i < map_inputs["SectionsTensorList"].size(); ++i) {
      sections.emplace_back(
          map_inputs["SectionsTensorList"].at(i)->GetConstData<int>()[0]);
    }
  }

  auto split_sections =
      builder::Const(input.GetBuilder(),
                     static_cast<void *>(sections.data()),
                     builder::Type({static_cast<int64_t>(sections.size())},
                                   builder::PrimitiveType::S32()));

  auto output = builder::Split(input, split_sections, axis, num);

  auto output_name_map = op->Outputs();
  std::vector<std::string> output_names = output_name_map["Out"];
  std::string output_names_attr(output_names[0]);
  for (size_t i = 1; i < output_names.size(); ++i) {
    output_names_attr += ";" + output_names[i];
  }
  output.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kSplit, INSENSITIVE, SplitEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
