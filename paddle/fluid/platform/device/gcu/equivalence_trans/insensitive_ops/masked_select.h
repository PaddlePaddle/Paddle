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
const char *const kMaskedSelect = "masked_select";
const char *const kMaskedSelectGrad = "masked_select_grad";

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, MaskedSelectEquivalenceTrans) {
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp mask = *(map_inputs["Mask"].at(0));
  auto mask_shape = mask.GetType().GetShape();
  builder::Type mask_int64_type(mask_shape, builder::PrimitiveType::S64());
  int64_t rank = x.GetType().GetRank();
  std::vector<int64_t> x_shape = x.GetType().GetShape();
  int64_t x_dims = 1;
  for (int64_t i = 0; i < rank; i++) {
    x_dims *= x_shape[i];
  }
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  auto mask_int64 = builder::Convert(mask, mask_int64_type);
  auto k = builder::ReduceSum(mask_int64, false, {dims});
  k = builder::Reshape(k, {{1}, builder::PrimitiveType::S64()});
  GcuOp zero_op = builder::ZerosLike(x);
  auto output = builder::Select(mask, x, zero_op);
  auto x_reshape =
      builder::Reshape(output, {{x_dims}, x.GetType().GetPrimitiveType()});
  auto top_out = builder::TopK(x_reshape, k, 0, false);
  auto result = builder::GetTupleElement(top_out, 0);
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               MaskedSelectGradEquivalenceTrans) {
  GcuOp x = *(map_inputs["X"].at(0));
  GcuOp mask = *(map_inputs["Mask"].at(0));
  GcuOp dout = *(map_inputs["Y@GRAD"].at(0));
  builder::Type scalar_type(x.GetType().GetPrimitiveType());
  auto mask_shape = mask.GetType().GetShape();
  builder::Type mask_int64_type(mask_shape, builder::PrimitiveType::S64());
  int64_t rank = x.GetType().GetRank();
  std::vector<int64_t> x_shape = x.GetType().GetShape();
  int64_t x_dims = 1;
  for (int64_t i = 0; i < rank; i++) {
    x_dims *= x_shape[i];
  }
  std::vector<int64_t> dims(rank);
  std::iota(dims.begin(), dims.end(), 0);
  auto mask_int64 = builder::Convert(mask, mask_int64_type);
  auto k = builder::ReduceSum(mask_int64, false, {dims});
  k = builder::Reshape(k, {{1}, builder::PrimitiveType::S64()});
  auto mask_reshape =
      builder::Reshape(mask, {{1, x_dims}, builder::PrimitiveType::S64()});
  auto x_reshape =
      builder::Reshape(x, {{1, x_dims}, x.GetType().GetPrimitiveType()});
  GcuOp zero_op = builder::ZerosLike(x_reshape);
  auto x_out = x_reshape * zero_op;
  auto top_out = builder::TopK(mask_reshape, k, 1);
  auto indices = builder::GetTupleElement(top_out, 1);
  auto result = builder::ScatterND(x_out, indices, dout);
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kMaskedSelect,
                           INSENSITIVE,
                           MaskedSelectEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kMaskedSelectGrad,
                           INSENSITIVE,
                           MaskedSelectGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
