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
#include "paddle/fluid/platform/device/gcu/utils/utils.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kOneHot = "one_hot";
const char *const kOneHotV2 = "one_hot_v2";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, OneHotEquivalenceTrans) {
  auto *op = node->Op();
  builder::Op label_op = *(map_inputs["X"].at(0));
  auto cls_index =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("depth")));
  auto out_dtype =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("dtype")));

  auto dtype = builder::PrimitiveType::NONE();
  if (out_dtype == framework::proto::VarType::FP32) {
    dtype = builder::PrimitiveType::F32();
  } else if (out_dtype == framework::proto::VarType::FP64) {
    dtype = builder::PrimitiveType::F64();
  } else if (out_dtype == framework::proto::VarType::INT16) {
    dtype = builder::PrimitiveType::S16();
  } else if (out_dtype == framework::proto::VarType::INT32) {
    dtype = builder::PrimitiveType::S32();
  } else if (out_dtype == framework::proto::VarType::INT64) {
    dtype = builder::PrimitiveType::S64();
  } else if (out_dtype == framework::proto::VarType::BOOL) {
    dtype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("fill_constant dtype: %d", out_dtype));
  }

  auto index_type = label_op.GetType();
  auto index_shape = index_type.GetShape();
  auto index_dtype = index_type.GetPrimitiveType();
  int64_t dim = static_cast<int64_t>(index_shape.size());
  if (cls_index < 0) {
    cls_index += (dim + 1);
  }
  int64_t axis = dim;
  std::vector<int64_t> output_shape(index_shape);
  output_shape.insert(output_shape.begin() + axis, cls_index);
  output_shape.erase(output_shape.begin() + axis - 1);
  auto output_index_type = builder::Type(output_shape, index_dtype);
  auto output_type = builder::Type(output_shape, dtype);
  auto iota = builder::Iota(gcu_builder, axis - 1, output_index_type);
  std::vector<int64_t> broadcast_dims(output_shape.size());
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  auto broadcast_indices =
      builder::BroadcastInDim(label_op, broadcast_dims, output_index_type);
  auto pred = builder::Compare(broadcast_indices, iota, "EQ");
  float one_data = 1.0f;
  void *data_ptr = static_cast<void *>(&one_data);
  auto one_value = builder::Const(gcu_builder, data_ptr, {{1}, dtype});
  float zero_data = 0.0f;
  data_ptr = static_cast<void *>(&zero_data);
  auto off_value = builder::Const(gcu_builder, data_ptr, {{1}, dtype});
  auto res = builder::Select(pred, one_value, off_value, output_type);

  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, OneHotV2EquivalenceTrans) {
  auto *op = node->Op();
  builder::Op input_op = *(map_inputs["X"].at(0));
  auto depth =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("depth")));
  auto dtype =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("dtype")));

  auto output_dtype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP32) {
    output_dtype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    output_dtype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::INT16) {
    output_dtype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    output_dtype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    output_dtype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    output_dtype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("one_hot_v2 dtype: %d", dtype));
  }

  const auto input_type = input_op.GetType();
  const auto input_dtype = input_type.GetPrimitiveType();
  const auto input_shape = input_type.GetShape();
  const int64_t input_size = input_type.GetSize();

  auto iota_shape = std::vector<int64_t>({input_size, depth});
  auto iota_type = builder::Type(iota_shape, input_dtype);
  auto iota_op = builder::Iota(gcu_builder, iota_shape.size() - 1, iota_type);

  auto flatten_op = builder::FlattenV2(input_op, 0, -1);
  auto reshape_op = builder::Reshape(flatten_op, {input_size, 1});
  auto broadcast_flatten_op =
      builder::BroadcastInDim(reshape_op, {0, 1}, iota_type);
  auto pred = builder::Equal(broadcast_flatten_op, iota_op);

  auto true_op = builder::OnesLike(broadcast_flatten_op, output_dtype);
  auto false_op = builder::ZerosLike(broadcast_flatten_op, output_dtype);
  auto res = builder::Select(pred, true_op, false_op);

  auto output_shape = input_shape;
  output_shape.emplace_back(depth);
  auto output_type = builder::Type(output_shape, output_dtype);
  res = builder::Reshape(res, output_type);

  return std::make_shared<GcuOp>(res);
}

EQUIVALENCE_TRANS_FUNC_REG(kOneHot, INSENSITIVE, OneHotEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kOneHotV2, INSENSITIVE, OneHotV2EquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
