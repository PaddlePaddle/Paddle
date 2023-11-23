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
const char *const kExpandAsV2 = "expand_as_v2";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ExpandAsV2EquivalenceTrans) {
  auto *op = node->Op();
  std::vector<int64_t> target_shape{};
  auto target_shape_ =
      PADDLE_GET_CONST(std::vector<int>, op->GetAttr("target_shape"));
  if (target_shape_.size() > 0) {
    target_shape =
        std::vector<int64_t>({target_shape_.begin(), target_shape_.end()});
  } else if (map_inputs.count("Y") != 0 && map_inputs["Y"].size() != 0) {
    GcuOp target = *(map_inputs["Y"].at(0));
    target_shape = target.GetType().GetShape();
  }

  auto input = *(map_inputs["X"].at(0));
  auto shape_op =
      builder::Const(input.GetBuilder(),
                     static_cast<void *>(target_shape.data()),
                     builder::Type({static_cast<int64_t>(target_shape.size())},
                                   builder::PrimitiveType::S64()));
  auto output_op = builder::Expand(input, shape_op);
  return std::make_shared<GcuOp>(output_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kExpandAsV2,
                           INSENSITIVE,
                           ExpandAsV2EquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
