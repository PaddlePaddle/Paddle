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
const char *const kGridSampler = "grid_sampler";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GridSamplerEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  builder::Op grid = *(map_inputs["Grid"].at(0));
  auto *op = node->Op();
  auto mode = PADDLE_GET_CONST(std::string, op->GetAttr("mode"));
  auto align_corners = PADDLE_GET_CONST(bool, op->GetAttr("align_corners"));
  auto padding_mode =
      PADDLE_GET_CONST(std::string, op->GetAttr("padding_mode"));
  std::map<std::string, int64_t> mode_map{{"nearest", 0}, {"bilinear", 1}};
  std::map<std::string, int64_t> padding_mode_map{
      {"zeros", 0}, {"border", 1}, {"reflection", 2}};
  auto input_rank = input.GetType().GetRank();

  int64_t batch_dimension = 0;
  int64_t feature_dimension = 3;
  std::vector<int64_t> spatial_dimensions(input_rank - 2);
  std::iota(spatial_dimensions.begin(), spatial_dimensions.end(), 1);
  builder::DimensionsLayout dimensionsLayout(
      batch_dimension, feature_dimension, spatial_dimensions);

  input = builder::Transpose(input, {0, 2, 3, 1});
  auto output = builder::GridSample(input,
                                    grid,
                                    dimensionsLayout,
                                    mode_map[mode],
                                    align_corners,
                                    padding_mode_map[padding_mode]);
  output = builder::Transpose(output, {0, 3, 1, 2});
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kGridSampler,
                           INSENSITIVE,
                           GridSamplerEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
