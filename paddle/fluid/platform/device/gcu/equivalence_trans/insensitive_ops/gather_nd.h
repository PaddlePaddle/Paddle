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
const char *const kGatherNd = "gather_nd";
const char *const kGatherNdGrad = "gather_nd_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GatherNdEquivalenceTrans) {
  GcuOp data = *(map_inputs["X"].at(0));
  GcuOp indices = *(map_inputs["Index"].at(0));
  auto data_shapes = data.GetType().GetShape();
  auto indices_shapes = indices.GetType().GetShape();
  int64_t input_rank = data_shapes.size();
  int64_t indices_rank = indices_shapes.size();

  int64_t indices_last_dim_size = indices_shapes.back();
  std::vector<int64_t> start_index_map;
  int64_t index_vector_dim = indices_rank - 1;
  for (int64_t i = 0; i < indices_last_dim_size; i++) {
    start_index_map.emplace_back(i);
  }
  auto collapsed_slice_dims = start_index_map;
  int64_t slices_rank = input_rank - indices_last_dim_size;
  int64_t batch_dims_size = indices_rank - 1;
  std::vector<int64_t> offset_dims;
  for (int64_t i = batch_dims_size; i < batch_dims_size + slices_rank; i++) {
    offset_dims.emplace_back(i);
  }
  auto dimension_numbers = builder::GatherDimensionNumbers(
      offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim);
  auto slice_sizes = std::vector<int64_t>(indices_last_dim_size, 1);
  for (int64_t i = indices_last_dim_size; i < input_rank; ++i)
    slice_sizes.push_back(data_shapes[i]);

  builder::Type scalar_type({builder::kUnknownRank},
                            data.GetType().GetPrimitiveType());
  builder::GatherDimensionNumbers gnums(
      offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim);
  auto result =
      builder::Gather(data, indices, gnums, slice_sizes, false, scalar_type);

  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GatherNdGradEquivalenceTrans) {
  GcuOp data = *(map_inputs["X"].at(0));
  GcuOp indices = *(map_inputs["Index"].at(0));
  GcuOp dout_op = *(map_inputs["Out@GRAD"].at(0));
  GcuOp data_op = builder::ZerosLike(data);

  GcuOp result_op = builder::ScatterND(data_op, indices, dout_op);
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kGatherNd, INSENSITIVE, GatherNdEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGatherNdGrad,
                           INSENSITIVE,
                           GatherNdGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
