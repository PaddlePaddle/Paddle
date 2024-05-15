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

const char* const kMeshgrid = "meshgrid";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MeshgridEquivalenceTrans) {
  std::vector<GcuOp> inputs;
  auto* op = node->Op();
  if (map_inputs["X"].size() <= 1 || map_inputs["X"].size() >= 7) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Excepted Tensor numbers between 2 and 6, but only received d% .",
        map_inputs["X"].size()));
  }
  for (size_t i = 0; i < map_inputs["X"].size(); ++i) {
    inputs.emplace_back(*(map_inputs["X"][i]));
  }
  auto output_name_map = op->Outputs();
  std::string output_names_attr = output_name_map["Out"][0];
  for (size_t i = 1; i < inputs.size(); ++i) {
    output_names_attr += ";" + output_name_map["Out"][i];
  }
  auto outputs = builder::MeshGrid(inputs);
  outputs.SetAttribute(kAttrOpOutVarName,
                       builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(outputs);
}

EQUIVALENCE_TRANS_FUNC_REG(kMeshgrid, INSENSITIVE, MeshgridEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
