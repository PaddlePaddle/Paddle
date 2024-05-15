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
const char *const kSum = "sum";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SumEquivalenceTrans) {
  std::vector<builder::Op> inputs;
  auto input_num = map_inputs["X"].size();
  for (size_t i = 0; i < input_num; ++i) {
    inputs.emplace_back(*(map_inputs["X"].at(i)));
  }
  size_t rank_0 = inputs[0].GetType().GetShape().size();
  bool all_inputs_same_rank =
      std::all_of(inputs.begin(), inputs.end(), [rank_0](builder::Op op) {
        return op.GetType().GetShape().size() == rank_0 && rank_0 == 4;
      });
  all_inputs_same_rank &= (running_mode == RunningMode::ADAPTIVE);
  if (all_inputs_same_rank) {
    for (size_t i = 0; i < input_num; i++) {
      inputs[i] = builder::Transpose(inputs[i], {0, 2, 3, 1});
    }
  }
  if (input_num == 1) {
    return all_inputs_same_rank
               ? std::make_shared<GcuOp>(builder::Transpose(
                     builder::Reshape(inputs[0], inputs[0].GetType()),
                     {0, 3, 1, 2}))
               : std::make_shared<GcuOp>(
                     builder::Reshape(inputs[0], inputs[0].GetType()));
  } else {
    builder::Op res;
    for (size_t i = 0; i < input_num; ++i) {
      if (inputs[i].GetType().GetSize() != 0) {
        if (!res.IsValid())
          res = inputs[i];
        else
          res = res + inputs[i];
      }
    }
    return all_inputs_same_rank
               ? std::make_shared<GcuOp>(builder::Transpose(res, {0, 3, 1, 2}))
               : std::make_shared<GcuOp>(res);
  }
}

EQUIVALENCE_TRANS_FUNC_REG(kSum, INSENSITIVE, SumEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
