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
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kFullLike = "fill_any_like";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, FullLikeEquivalenceTrans) {
  auto *op = node->Op();
  auto input_x = *(map_inputs["X"].at(0));
  auto out_dtype = PADDLE_GET_CONST(int, op->GetAttr("dtype"));
  auto full_value = PADDLE_GET_CONST(float, op->GetAttr("value"));

  auto ptype = builder::PrimitiveType::NONE();
  if (out_dtype == -1) {
    ptype = input_x.GetType().GetPrimitiveType();
  } else if (out_dtype == framework::proto::VarType::FP32) {
    ptype = builder::PrimitiveType::F32();
  } else if (out_dtype == framework::proto::VarType::FP64) {
    ptype = builder::PrimitiveType::F64();
  } else if (out_dtype == framework::proto::VarType::INT16) {
    ptype = builder::PrimitiveType::S16();
  } else if (out_dtype == framework::proto::VarType::INT32) {
    ptype = builder::PrimitiveType::S32();
  } else if (out_dtype == framework::proto::VarType::INT64) {
    ptype = builder::PrimitiveType::S64();
  } else if (out_dtype == framework::proto::VarType::BOOL) {
    ptype = builder::PrimitiveType::PRED();
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("fill_any_like dtype: %d", out_dtype));
  }
  auto full_value_op = builder::FullLike(input_x, full_value, ptype);
  return std::make_shared<GcuOp>(full_value_op);
}
EQUIVALENCE_TRANS_FUNC_REG(kFullLike, INSENSITIVE, FullLikeEquivalenceTrans);
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
