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
const char *const kGather = "gather";
const char *const kGatherGrad = "gather_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GatherEquivalenceTrans) {
  auto *op = node->Op();
  GcuOp data = *(map_inputs["X"].at(0));
  GcuOp indices = *(map_inputs["Index"].at(0));
  auto data_shape = data.GetType().GetShape();
  std::vector<int64_t> indices_shape = indices.GetType().GetShape();

  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));
  if (map_inputs.count("Axis") != 0) {
    axis = map_inputs["Axis"].at(0)->GetConstData<int>()[0];
  } else {
    axis = axis < 0 ? axis + data.GetType().GetShape().size() : axis;
  }

  std::vector<int64_t> offset_dim;
  std::vector<int64_t> collapsed_slice_dims;
  std::vector<int64_t> start_index_map;
  int64_t index_vector_dim = 1;
  std::vector<int64_t> slice_size(data_shape);

  std::vector<int64_t> out_shape(data_shape);
  out_shape[axis] = indices_shape[0];

  builder::Type scalar_type(out_shape, data.GetType().GetPrimitiveType());
  for (int64_t i = 0; i < axis; i++) {
    offset_dim.emplace_back(i);
  }

  for (size_t i = axis + 1; i < data_shape.size(); i++) {
    offset_dim.emplace_back(i);
  }
  collapsed_slice_dims.emplace_back(axis);
  start_index_map.emplace_back(axis);
  slice_size[axis] = 1;

  builder::GatherDimensionNumbers gnums(
      offset_dim, collapsed_slice_dims, start_index_map, index_vector_dim);
  auto result =
      builder::Gather(data, indices, gnums, slice_size, false, scalar_type);

  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, GatherGradEquivalenceTrans) {
  GcuOp data = *(map_inputs["X"].at(0));
  GcuOp index = *(map_inputs["Index"].at(0));
  GcuOp source = *(map_inputs["Out@GRAD"].at(0));
  auto *op = node->Op();
  auto axis = PADDLE_GET_CONST(int, op->GetAttr("axis"));

  if (map_inputs.count("Axis") != 0) {
    axis = map_inputs["Axis"].at(0)->GetConstData<int>()[0];
  } else {
    axis = axis < 0 ? axis + data.GetType().GetShape().size() : axis;
  }
  // create buffer
  auto input = builder::ZerosLike(data);
  // create scatter params
  auto input_ptype = input.GetType().GetPrimitiveType();
  auto input_shape = input.GetType().GetShape();
  int64_t input_rank = input_shape.size();

  builder::ScatterDimensionNumbers scatter_dnums;
  scatter_dnums.set_index_vector_dim(input_rank);
  for (int64_t i = 0; i < input_rank; ++i) {
    scatter_dnums.add_inserted_window_dims(i);
    scatter_dnums.add_scatter_dims_to_operand_dims(i);
  }

  std::vector<int64_t> update_window_dims;
  std::vector<int64_t> inserted_window_dims;
  std::vector<int64_t> scatter_dims_to_operand_dims;
  int64_t index_vector_dim = 1;
  for (int64_t i = 1; i < input_rank; i++) {
    update_window_dims.emplace_back(i);
  }
  inserted_window_dims.emplace_back(0);
  scatter_dims_to_operand_dims.emplace_back(0);

  builder::ScatterDimensionNumbers dim_numbers(update_window_dims,
                                               inserted_window_dims,
                                               scatter_dims_to_operand_dims,
                                               index_vector_dim);
  auto tmp_region_list = CreateBindingFunc(
      input.GetBuilder(), {BindingFuncType::ADD}, {input_ptype});
  std::vector<const char *> region_list;
  for (auto &region : tmp_region_list) region_list.push_back(region.c_str());
  GcuOp res;
  if (axis == 0) {
    res = builder::Scatter(input, index, source, dim_numbers, region_list);
  } else {
    // transpose scatter axis to 0
    std::vector<int64_t> transpose_permu;
    std::vector<int64_t> ori_permu;
    for (int64_t i = 0; i < input_rank; i++) {
      if (i == axis) {
        transpose_permu.insert(transpose_permu.begin(), i);
        ori_permu.push_back(0);
      } else {
        transpose_permu.push_back(i);
        if (i < axis) {
          ori_permu.push_back(i + 1);
        } else {
          ori_permu.push_back(i);
        }
      }
    }
    GcuOp input_trans;
    GcuOp out_grad_trans;
    GcuOp source_trans;
    input_trans = builder::Transpose(input, transpose_permu);
    source_trans = builder::Transpose(source, transpose_permu);
    auto res_trans = builder::Scatter(
        input_trans, index, source_trans, dim_numbers, region_list);
    // transpose result back
    res = builder::Transpose(res_trans, ori_permu);
  }

  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kGather, INSENSITIVE, GatherEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kGatherGrad,
                           INSENSITIVE,
                           GatherGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
