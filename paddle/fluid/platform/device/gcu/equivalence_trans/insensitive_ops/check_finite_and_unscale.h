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
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace paddle {
namespace platform {
namespace gcu {
const char* const kCheckFiniteAndUnscale = "check_finite_and_unscale";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               CheckFiniteAndUnscaleEquivalenceTrans) {
  auto input_num = map_inputs["X"].size();
  PADDLE_ENFORCE_EQ(input_num,
                    1,
                    platform::errors::Unimplemented(
                        "for op %s, gcu does not support len(X) > 1.",
                        kCheckFiniteAndUnscale));
  auto input = *(map_inputs["X"].at(0));
  auto scale = *(map_inputs["Scale"].at(0));

  auto is_infinite = builder::Not(builder::IsFinite(input));
  auto found_inf = builder::ReduceSum(is_infinite, false);
  found_inf = builder::Reshape(
      found_inf, builder::Type({1}, found_inf.GetType().GetPrimitiveType()));

  auto out = builder::Select(found_inf, input, input / scale);

  auto* op = node->Op();
  auto output_name_map = op->Outputs();
  std::string output_names_attr =
      output_name_map["Out"][0] + ";" + output_name_map["FoundInfinite"][0];

  auto result = builder::Tuple({out, found_inf});
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kCheckFiniteAndUnscale,
                           INSENSITIVE,
                           CheckFiniteAndUnscaleEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
