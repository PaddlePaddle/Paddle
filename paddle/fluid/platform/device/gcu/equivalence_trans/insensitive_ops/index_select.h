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

#include <algorithm>
#include <memory>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char* const kIndexSelect = "index_select";
const char* const kIndexSelectGrad = "index_select_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, IndexSelectEquivalenceTrans) {
  auto* op = node->Op();
  GcuOp inputs = *(map_inputs["X"].at(0));
  GcuOp index = *(map_inputs["Index"].at(0));
  auto dim = PADDLE_GET_CONST(int32_t, op->GetAttr("dim"));

  int32_t batch_dims = 0;
  int32_t input_rank = inputs.GetType().GetShape().size();
  auto input_shape = inputs.GetType().GetShape();
  // to do: check dim
  if (dim < batch_dims) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("Gather dim must be greater than or "
                                          "equal to the number of batch dims"));
  }
  // to do: add index_ptype convert
  // auto index_ptype = index.GetType().GetPrimitiveType();
  std::vector<int64_t> slice_sizes = input_shape;
  builder::GatherDimensionNumbers gather_dnums;
  gather_dnums.set_index_vector_dim(index.GetType().GetShape().size());

  for (int32_t i = 0; i < input_rank; ++i) {
    if (i < batch_dims || i == dim) {
      slice_sizes[i] = std::min<int64_t>(slice_sizes[i], 1);
      gather_dnums.add_collapsed_slice_dims(i);
      gather_dnums.add_start_index_map(i);
    } else {
      if (i < dim) {
        gather_dnums.add_offset_dims(i);
      } else {
        gather_dnums.add_offset_dims(i + gather_dnums.get_index_vector_dim() -
                                     (1 + batch_dims));
      }
    }
  }
  GcuOp result = builder::Gather(inputs, index, gather_dnums, slice_sizes);
  return std::make_shared<GcuOp>(result);
}

static GcuOp BuildBroadcast(GcuOp operand,
                            const std::vector<int64_t>& broadcast_sizes) {
  auto op_shape = operand.GetType().GetShape();
  std::vector<int64_t> dimensions(op_shape.size() + broadcast_sizes.size());
  std::copy(broadcast_sizes.begin(), broadcast_sizes.end(), dimensions.begin());
  std::copy(op_shape.begin(),
            op_shape.end(),
            dimensions.begin() + broadcast_sizes.size());

  int operand_rank = op_shape.size();
  std::vector<int64_t> broadcast_dimensions(operand_rank);
  for (auto i = 0; i < operand_rank; ++i) {
    broadcast_dimensions[i] = i + dimensions.size() - operand_rank;
  }
  return builder::BroadcastInDim(
      operand,
      broadcast_dimensions,
      {dimensions, operand.GetType().GetPrimitiveType()});
}

static GcuOp CreateIndexAlongDim(
    const GcuOp& buffer,
    int64_t dim,
    const GcuOp& index,
    const GcuOp& value,
    bool broadcast_value_to_index,
    const BindingFuncType& func_type = BindingFuncType::ADD) {
  auto buffer_shape = buffer.GetType().GetShape();
  auto buffer_ptype = buffer.GetType().GetPrimitiveType();
  int64_t buffer_rank = buffer_shape.size();

  builder::ScatterDimensionNumbers dim_numbers;
  dim_numbers.set_index_vector_dim(1);
  for (int64_t window_dim = 0; window_dim < buffer_rank; ++window_dim) {
    if (window_dim != dim) {
      dim_numbers.add_update_window_dims(window_dim);
    } else {
      dim_numbers.add_inserted_window_dims(window_dim);
      dim_numbers.add_scatter_dims_to_operand_dims(window_dim);
    }
  }

  // Broadcast the value to the right shape required by scatter.
  auto value_shape = value.GetType().GetShape();
  auto value_ptype = value.GetType().GetPrimitiveType();

  GcuOp updates = value;
  // to do : conver type
  // if (buffer_ptype != value_ptype) {
  //   updates = ConvertTo(updates, buffer_ptype);
  // }
  if (broadcast_value_to_index) {
    auto index_shape = index.GetType().GetShape();
    auto _shape = buffer.GetType().GetShape();
    std::vector<int64_t> update_dimensions = buffer_shape;
    update_dimensions[dim] = index_shape[0];
    updates = BuildBroadcast(updates, update_dimensions);
  }
  auto tmp_region_list =
      CreateBindingFunc(buffer.GetBuilder(), {func_type}, {buffer_ptype});
  std::vector<const char*> region_list;
  for (auto& region : tmp_region_list) region_list.push_back(region.c_str());
  return builder::Scatter(buffer, index, updates, dim_numbers, region_list);
}

static GcuOp BuildIndexAdd(const GcuOp& buffer,
                           int64_t dim,
                           const GcuOp& index,
                           const GcuOp& source) {
  return CreateIndexAlongDim(buffer,
                             dim,
                             index,
                             source,
                             /*broadcast_value_to_index=*/false,
                             BindingFuncType::ADD);
}

template <typename T>
GcuOp GetZerosOp(const GcuOp& src_op) {
  std::vector<T> zeros_data;
  auto input_shape = src_op.GetType().GetShape();
  int32_t input_rank = input_shape.size();
  int32_t data_num = 1;
  for (int32_t i = 0; i < input_rank; i++) {
    data_num *= input_shape[i];
  }
  for (int32_t i = 0; i < data_num; i++) {
    zeros_data.push_back(0);
  }
  GcuOp buffer = builder::Const(src_op.GetBuilder(),
                                static_cast<void*>(zeros_data.data()),
                                src_op.GetType());
  return buffer;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               IndexSelectGradEquivalenceTrans) {
  auto* op = node->Op();
  GcuOp inputs = *(map_inputs["X"].at(0));
  GcuOp index = *(map_inputs["Index"].at(0));
  GcuOp out_grad = *(map_inputs["Out@GRAD"].at(0));
  auto dim = PADDLE_GET_CONST(int32_t, op->GetAttr("dim"));
  // to do: index ensure rank1
  auto primitive_type = inputs.GetType().GetPrimitiveType();
  GcuOp buffer;
  if (primitive_type == builder::PrimitiveType::F32()) {
    buffer = GetZerosOp<float>(inputs);
  } else if (primitive_type == builder::PrimitiveType::F64()) {
    buffer = GetZerosOp<double>(inputs);
  } else if (primitive_type == builder::PrimitiveType::S32()) {
    buffer = GetZerosOp<int32_t>(inputs);
  } else if (primitive_type == builder::PrimitiveType::S64()) {
    buffer = GetZerosOp<int64_t>(inputs);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unable to lower scalar %d of type", primitive_type));
  }

  int32_t input_rank = inputs.GetType().GetShape().size();
  GcuOp inputs_grad;
  if (dim == 0) {
    inputs_grad = BuildIndexAdd(buffer, dim, index, out_grad);
  } else {
    // transpose selected dim to 0
    std::vector<int64_t> transpose_permu;
    std::vector<int64_t> ori_permu;
    for (int64_t i = 0; i < input_rank; i++) {
      if (i == dim) {
        transpose_permu.insert(transpose_permu.begin(), i);
        ori_permu.push_back(0);
      } else {
        transpose_permu.push_back(i);
        if (i < dim) {
          ori_permu.push_back(i + 1);
        } else {
          ori_permu.push_back(i);
        }
      }
    }
    GcuOp buffer_trans;
    GcuOp out_grad_trans;
    GcuOp inputs_grad_trans;
    buffer_trans = builder::Transpose(buffer, transpose_permu);
    out_grad_trans = builder::Transpose(out_grad, transpose_permu);
    inputs_grad_trans = BuildIndexAdd(buffer_trans, 0, index, out_grad_trans);
    // transpose dim 0 to original selected dim
    inputs_grad = builder::Transpose(inputs_grad_trans, ori_permu);
  }
  return std::make_shared<GcuOp>(inputs_grad);
}

EQUIVALENCE_TRANS_FUNC_REG(kIndexSelect,
                           INSENSITIVE,
                           IndexSelectEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kIndexSelectGrad,
                           INSENSITIVE,
                           IndexSelectGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
