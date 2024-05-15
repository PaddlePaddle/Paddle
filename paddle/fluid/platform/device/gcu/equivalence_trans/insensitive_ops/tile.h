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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char* const kTile = "tile";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TileEquivalenceTrans) {
  auto input = *(map_inputs["X"].at(0));
  auto repeat_op = [&]() {
    auto op = node->Op();
    if (map_inputs.count("RepeatTimes") != 0) {
      auto repeat_op = *(map_inputs.at("RepeatTimes").at(0));
      return repeat_op;
    } else if (map_inputs.count("repeat_times_tensor")) {
      auto repeat_tensor_list = map_inputs.at("repeat_times_tensor");
      std::vector<builder::Op> repeat_ops;
      std::vector<int64_t> new_shape = {1};
      for (const auto& repeat_tensor : repeat_tensor_list) {
        repeat_ops.emplace_back(builder::Reshape(*repeat_tensor, new_shape));
      }
      return builder::Concatenate(repeat_ops, 0);
    } else {
      auto repeat_times =
          PADDLE_GET_CONST(std::vector<int>, op->GetAttr("repeat_times"));
      return builder::Const(
          input.GetBuilder(),
          repeat_times,
          builder::Type({static_cast<int64_t>(repeat_times.size())},
                        builder::PrimitiveType::S32()));
    }
  }();
  const int64_t input_rank = input.GetType().GetRank();
  const int64_t repeat_size = repeat_op.GetType().GetSize();
  if (input_rank < repeat_size) {
    int64_t diff = repeat_size - input_rank;
    std::vector<int64_t> new_shape(diff, 1);
    auto input_shape = input.GetType().GetShape();
    new_shape.insert(new_shape.end(), input_shape.begin(), input_shape.end());
    input = builder::Reshape(input, new_shape);
  } else if (input_rank > repeat_size) {
    int64_t diff = input_rank - repeat_size;
    std::vector<int32_t> diff_repeat(diff, 1);
    auto diff_op =
        builder::Const(gcu_builder,
                       diff_repeat,
                       builder::Type({diff}, builder::PrimitiveType::S32()));
    std::vector<builder::Op> repeat_ops = {diff_op, repeat_op};
    repeat_op = builder::Concatenate(repeat_ops, 0);
  }
  PADDLE_ENFORCE_EQ(
      input.GetType().GetRank(),
      repeat_op.GetType().GetSize(),
      platform::errors::InvalidArgument(
          "The rank (%d) of the input for 'tile' must equal to the "
          "size (%d) of the repeat.",
          input.GetType().GetRank(),
          repeat_op.GetType().GetSize()));

  return std::make_shared<GcuOp>(builder::Tile(input, repeat_op));
}

EQUIVALENCE_TRANS_FUNC_REG(kTile, INSENSITIVE, TileEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
