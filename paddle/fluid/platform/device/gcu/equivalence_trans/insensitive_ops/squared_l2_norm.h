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
const char *const kSquaredL2Norm = "squared_l2_norm";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               SquaredL2NormEquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  auto const_2 = builder::FullLike(input, 2.0);
  auto pow_input = builder::Pow(input, const_2);
  auto reduce_sum = builder::ReduceSum(pow_input, false);
  std::vector<int64_t> new_shape = {1};
  auto result = builder::Reshape(reduce_sum, new_shape);
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kSquaredL2Norm,
                           INSENSITIVE,
                           SquaredL2NormEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
