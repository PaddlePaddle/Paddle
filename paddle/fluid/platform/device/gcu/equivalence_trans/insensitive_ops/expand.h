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
const char *const kExpandV2 = "expand_v2";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, ExpandV2EquivalenceTrans) {
  GcuOp input = *(map_inputs["X"].at(0));
  auto *op = node->Op();
  auto shape = PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));

  GcuOp shape_op;
  if (map_inputs.count("Shape") != 0) {
    shape_op = *(map_inputs["Shape"].at(0));
  } else if (map_inputs.count("expand_shapes_tensor") != 0) {
    std::vector<GcuOp> shape_list;
    for (size_t i = 0; i < map_inputs["expand_shapes_tensor"].size(); ++i) {
      auto dim_op = *(map_inputs["expand_shapes_tensor"].at(i));
      const int64_t rank = dim_op.GetType().GetRank();
      PADDLE_ENFORCE_LE(
          rank,
          1,
          platform::errors::InvalidArgument(
              "ExpandV2 expand_shapes_tensor's rank must <= 1, but got: %d",
              rank));
      if (rank == 0) dim_op = builder::Reshape(dim_op, {1});
      shape_list.emplace_back(dim_op);
    }
    shape_op = builder::Concatenate(shape_list, 0);
  } else if (!shape.empty()) {
    shape_op =
        builder::Const(input.GetBuilder(),
                       static_cast<void *>(shape.data()),
                       builder::Type({static_cast<int64_t>(shape.size())},
                                     builder::PrimitiveType::S32()));
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("Unsupported ExpandV2 without shape"));
  }

  auto out_op = builder::Expand(input, shape_op);
  return std::make_shared<GcuOp>(out_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kExpandV2, INSENSITIVE, ExpandV2EquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
