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
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char* const kLogicalAnd = "logical_and";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, LogicalAndEquivalenceTrans) {
  std::vector<GcuOp> inputs;
  if (map_inputs.count("X") != 0) {
    inputs.push_back(*(map_inputs["X"].at(0)));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [X] gcu op"));
  }
  if (map_inputs.count("Y") != 0) {
    inputs.push_back(*(map_inputs["Y"].at(0)));
  } else {
    PADDLE_ENFORCE_EQ(
        true, false, platform::errors::NotFound("lack of [Y] gcu op"));
  }

  auto lhs_shape = inputs[0].GetType().GetShape();
  auto rhs_shape = inputs[1].GetType().GetShape();

  if (lhs_shape != rhs_shape) {
    PADDLE_THROW(platform::errors::Fatal("lhs_shape not equal."));
  }

  builder::Type output_type(inputs[0].GetType().GetShape(),
                            builder::PrimitiveType::PRED());
  if (inputs[0].GetType().GetPrimitiveType() !=
      builder::PrimitiveType::PRED()) {
    inputs[0] = builder::Convert(inputs[0], output_type);
  }
  if (inputs[1].GetType().GetPrimitiveType() !=
      builder::PrimitiveType::PRED()) {
    inputs[1] = builder::Convert(inputs[1], output_type);
  }
  auto out = builder::And(inputs[0], inputs[1]);
  return std::make_shared<GcuOp>(out);
}

EQUIVALENCE_TRANS_FUNC_REG(kLogicalAnd,
                           INSENSITIVE,
                           LogicalAndEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
