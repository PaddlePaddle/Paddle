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
#include <utility>
#include <vector>
#include "paddle/fluid/platform/device/gcu/equivalence_trans/utils.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kLookupTableV2 = "lookup_table_v2";
const char *const kLookupTableV2Grad = "lookup_table_v2_grad";
constexpr int64_t kNoPadding = -1;

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               LookupTableV2EquivalenceTrans) {
  auto *op = node->Op();
  auto ids = *(map_inputs["Ids"].at(0));
  auto W = *(map_inputs["W"].at(0));
  auto padding_idx = static_cast<int64_t>(
      PADDLE_GET_CONST(int64_t, op->GetAttr("padding_idx")));
  builder::Op table_op = W;
  if (padding_idx != kNoPadding) {
    auto index_op = builder::Const(
        gcu_builder, padding_idx, builder::Type(builder::PrimitiveType::S64()));
    index_op = builder::Reshape(index_op, {1, 1});
    auto update_op = builder::Const(
        gcu_builder, 0, builder::Type(W.GetType().GetPrimitiveType()));
    if (update_op.GetType().GetRank() == 0) {
      update_op = builder::Reshape(update_op, {1});
    }
    auto scalar_type = builder::Type({1, W.GetType().GetShape()[1]},
                                     W.GetType().GetPrimitiveType());
    update_op = builder::BroadcastInDim(update_op, {1}, scalar_type);
    size_t index_vector_dim = 1;
    std::vector<int64_t> inserted_window_dims = {0};
    std::vector<int64_t> scatter_dims_to_operand_dims = {0};
    std::vector<int64_t> update_window_dims = {1};
    builder::ScatterDimensionNumbers dim_numbers(update_window_dims,
                                                 inserted_window_dims,
                                                 scatter_dims_to_operand_dims,
                                                 index_vector_dim);
    auto tmp_region_list = CreateBindingFunc(W.GetBuilder(),
                                             {BindingFuncType::IDENTITY},
                                             {W.GetType().GetPrimitiveType()});
    std::vector<const char *> region_list;
    for (auto &region : tmp_region_list) region_list.push_back(region.c_str());
    table_op =
        builder::Scatter(W, index_op, update_op, dim_numbers, region_list);
  }
  auto ids_shape = ids.GetType().GetShape();
  int64_t ids_dims = static_cast<int64_t>(ids_shape.size());
  int64_t index_ids_dim = 1;
  for (int i = 0; i < ids_dims; ++i) {
    index_ids_dim *= ids_shape[i];
  }
  auto ids_reshape = builder::Reshape(ids, {index_ids_dim});
  std::vector<int64_t> offset_dims = {1};
  std::vector<int64_t> collapsed_slice_dims = {0};
  std::vector<int64_t> start_index_map = {0};
  int64_t index_vector_dim = 1;
  auto dimension_numbers = builder::GatherDimensionNumbers(
      offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim);
  int64_t embeding_num = table_op.GetType().GetShape()[1];
  std::vector<int64_t> slice_sizes = {index_vector_dim, embeding_num};
  auto output =
      builder::Gather(table_op, ids_reshape, dimension_numbers, slice_sizes);
  std::vector<int64_t> output_shape = ids_shape;
  output_shape.emplace_back(embeding_num);
  output = builder::Reshape(output, output_shape);
  return std::make_shared<GcuOp>(output);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               LookUpTableV2GradEquivalenceTrans) {
  auto *op = node->Op();
  auto ids = *(map_inputs["Ids"].at(0));
  auto W = *(map_inputs["W"].at(0));
  auto out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto padding_idx = static_cast<int64_t>(
      PADDLE_GET_CONST(int64_t, op->GetAttr("padding_idx")));
  auto ids_shape = ids.GetType().GetShape();
  auto out_shape = out_grad.GetType().GetShape();
  auto table_grad = builder::ZerosLike(W, W.GetType().GetPrimitiveType());
  int64_t ids_dims = static_cast<int64_t>(ids_shape.size());
  int64_t index_ids_dim = 1;
  for (int i = 0; i < ids_dims; ++i) {
    index_ids_dim *= ids_shape[i];
  }
  auto ids_reshape = builder::Reshape(ids, {index_ids_dim, 1});
  auto padding_op = builder::FullLike(ids_reshape, padding_idx);
  auto compare_op = builder::Compare(ids_reshape, padding_op, "NE");

  std::vector<int64_t> bdst_shape = {ids_reshape.GetType().GetShape()[0],
                                     W.GetType().GetShape()[1]};
  auto scalar_type =
      builder::Type(bdst_shape, compare_op.GetType().GetPrimitiveType());
  auto ids_bdst = builder::BroadcastInDim(compare_op, {0, 1}, scalar_type);
  auto out_reshape = builder::Reshape(out_grad, ids_bdst.GetType().GetShape());

  auto zero_op =
      builder::ZerosLike(out_reshape, out_reshape.GetType().GetPrimitiveType());

  auto select_op = builder::Select(compare_op, out_reshape, zero_op);
  int64_t index_vector_dim = 1;
  std::vector<int64_t> inserted_window_dims = {0};
  std::vector<int64_t> scatter_dims_to_operand_dims = {0};
  std::vector<int64_t> update_window_dims = {1};
  builder::ScatterDimensionNumbers dim_numbers(update_window_dims,
                                               inserted_window_dims,
                                               scatter_dims_to_operand_dims,
                                               index_vector_dim);
  auto tmp_region_list = CreateBindingFunc(
      W.GetBuilder(), {BindingFuncType::ADD}, {W.GetType().GetPrimitiveType()});
  std::vector<const char *> region_list;
  for (auto &region : tmp_region_list) region_list.push_back(region.c_str());
  auto output = builder::Scatter(
      table_grad, ids_reshape, select_op, dim_numbers, region_list);
  return std::make_shared<GcuOp>(output);
}

EQUIVALENCE_TRANS_FUNC_REG(kLookupTableV2,
                           INSENSITIVE,
                           LookupTableV2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kLookupTableV2Grad,
                           INSENSITIVE,
                           LookUpTableV2GradEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
