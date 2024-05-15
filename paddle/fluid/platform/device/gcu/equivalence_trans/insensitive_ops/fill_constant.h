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

#include <limits>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kFillConstant = "fill_constant";
const char *const kFillConstantBatchSizeLike = "fill_constant_batch_size_like";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FillConstantEquivalenceTrans) {
  builder::Op scalar;
  if (map_inputs.count("ValueTensor") != 0) {
    scalar = *(map_inputs["ValueTensor"].at(0));
  }
  auto *op = node->Op();
  std::vector<int64_t> shape;
  auto shape_attr = op->GetAttr("shape");
  if (phi::enforce::demangle(shape_attr.type().name()) ==
      "std::vector<int, std::allocator<int> >") {
    auto origin_shape =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
    shape = std::move(
        std::vector<int64_t>(origin_shape.begin(), origin_shape.end()));
  } else {
    shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  }

  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto ptype = builder::PrimitiveType::NONE();
  auto value = PADDLE_GET_CONST(float, op->GetAttr("value"));
  if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("fill_constant dtype: %d", dtype));
  }
  if (!scalar.IsValid()) {
    scalar = builder::Const(gcu_builder, value, builder::Type(shape, ptype));
  }
  return std::make_shared<GcuOp>(scalar);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               FillConstantBatchSizeLikeEquivalenceTrans) {
  auto *op = node->Op();
  auto input = *(map_inputs["Input"].at(0));
  std::vector<int64_t> shape;
  auto shape_attr = op->GetAttr("shape");
  if (phi::enforce::demangle(shape_attr.type().name()) ==
      "std::vector<int, std::allocator<int> >") {
    auto origin_shape =
        PADDLE_GET_CONST(std::vector<int>, op->GetAttr("shape"));
    shape = std::move(
        std::vector<int64_t>(origin_shape.begin(), origin_shape.end()));
  } else {
    shape = PADDLE_GET_CONST(std::vector<int64_t>, op->GetAttr("shape"));
  }
  auto x_batch_size_dim = PADDLE_GET_CONST(int, op->GetAttr("input_dim_idx"));
  auto out_batch_size_dim =
      PADDLE_GET_CONST(int, op->GetAttr("output_dim_idx"));
  auto input_shape = input.GetType().GetShape();
  shape[out_batch_size_dim] = input_shape[x_batch_size_dim];

  auto dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto ptype = builder::PrimitiveType::NONE();
  if (dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("fill_constant dtype: %d", dtype));
  }
  double value = PADDLE_GET_CONST(float, op->GetAttr("value"));
  auto str_value = PADDLE_GET_CONST(std::string, op->GetAttr("str_value"));
  if (!str_value.empty()) {
    if (str_value == "inf") {
      value = std::numeric_limits<float>::infinity();
    } else if (str_value == "-inf") {
      value = -std::numeric_limits<float>::infinity();
    } else if (str_value == "nan") {
      value = std::numeric_limits<float>::quiet_NaN();
    } else {
      std::stringstream convert_stream(str_value);
      convert_stream >> value;
    }
  }
  auto fill_like_op = builder::FullLike(input, value, ptype, shape);
  return std::make_shared<GcuOp>(fill_like_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kFillConstantBatchSizeLike,
                           INSENSITIVE,
                           FillConstantBatchSizeLikeEquivalenceTrans);

EQUIVALENCE_TRANS_FUNC_REG(kFillConstant,
                           INSENSITIVE,
                           FillConstantEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
