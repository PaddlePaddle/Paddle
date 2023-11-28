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
const char *const kTrilTriu = "tril_triu";
const char *const kTrilTriuGrad = "tril_triu_grad";
const char *const kTril = "tril";
const char *const kTrilGrad = "tril_grad";
const char *const kTriu = "triu";
const char *const kTriuGrad = "triu_grad";

// Comment: Temp workaround to support trilu with input which has nan and inf
// value. As Hlir's Trilu use `mul(input, const upper/lower triangle)` to
// implment trilu, when input has NAN or INF value, the `mul` may have wrong
// result, here we use `select` to support trilu.
inline builder::Op TriluWrapper(builder::Op input,
                                bool upper = true,
                                int64_t diagonal = 0,
                                builder::Type result_type = builder::Type()) {
  auto input_shape = input.GetType().GetShape();
  auto input_rank = input.GetType().GetRank();

  const int64_t H = input_shape[input_rank - 2];
  const int64_t W = input_shape[input_rank - 1];
  const int64_t B = [&]() {
    int64_t batch = 1;
    for (int64_t i = 0; i < input_rank - 2; ++i) batch *= input_shape[i];
    return batch;
  }();

  std::vector<int8_t> tmp(H * W);
  for (int64_t i = 0; i < H * W; ++i) {
    const int64_t row = (i / W) % H;
    const int64_t col = i % W;
    const bool mask = !upper ? (col - row > diagonal) : (col - row < diagonal);
    tmp[i] = mask ? 0 : 1;
  }

  std::vector<int8_t> cond_data(B * H * W);
  for (int64_t i = 0; i < B; ++i) {
    std::memcpy(
        cond_data.data() + i * H * W, tmp.data(), H * W * sizeof(int8_t));
  }

  const std::vector<int64_t> new_shape = {B, H, W};
  auto cond_op =
      builder::Const(input.GetBuilder(),
                     cond_data,
                     builder::Type(new_shape, builder::PrimitiveType::S8()));
  cond_op = builder::Convert(
      cond_op, builder::Type(new_shape, builder::PrimitiveType::PRED()));
  auto true_op = builder::Reshape(input, new_shape);
  auto false_op = builder::ZerosLike(true_op);
  auto out_op = builder::Select(cond_op, true_op, false_op);
  out_op = builder::Reshape(out_op, input_shape);
  return out_op;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TrilTriuEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto lower = PADDLE_GET_CONST(bool, op->GetAttr("lower"));
  auto out_op = TriluWrapper(input, !lower, diagonal);
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TrilTriuGradEquivalenceTrans) {
  builder::Op grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto lower = PADDLE_GET_CONST(bool, op->GetAttr("lower"));
  auto grad_in = TriluWrapper(grad_out, !lower, diagonal);
  return std::make_shared<GcuOp>(grad_in);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TrilEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto out_op = TriluWrapper(input, false, diagonal);
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TrilGradEquivalenceTrans) {
  builder::Op grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto grad_in = TriluWrapper(grad_out, false, diagonal);
  return std::make_shared<GcuOp>(grad_in);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TriuEquivalenceTrans) {
  builder::Op input = *(map_inputs["X"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto out_op = TriluWrapper(input, true, diagonal);
  return std::make_shared<GcuOp>(out_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, TriuGradEquivalenceTrans) {
  builder::Op grad_out = *(map_inputs["Out@GRAD"].at(0));
  auto op = node->Op();
  auto diagonal = PADDLE_GET_CONST(int, op->GetAttr("diagonal"));
  auto grad_in = TriluWrapper(grad_out, true, diagonal);
  return std::make_shared<GcuOp>(grad_in);
}

EQUIVALENCE_TRANS_FUNC_REG(kTrilTriu, INSENSITIVE, TrilTriuEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTrilTriuGrad,
                           INSENSITIVE,
                           TrilTriuGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTril, INSENSITIVE, TrilEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTrilGrad, INSENSITIVE, TrilGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTriu, INSENSITIVE, TriuEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kTriuGrad, INSENSITIVE, TriuGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
