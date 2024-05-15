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
const char* const kSoftmax = "softmax";
const char* const kSoftmaxGrad = "softmax_grad";

template <typename T>
builder::Op ScalarOp(std::shared_ptr<builder::Builder> builder,
                     T value,
                     builder::Type type) {
#define HLIR_CONST(T)                                                    \
  do {                                                                   \
    auto v = static_cast<T>(value);                                      \
    std::vector<T> vec(n, v);                                            \
    hop = builder::Const(builder, static_cast<void*>(vec.data()), type); \
  } while (0);

  builder::Op hop;
  auto shape = type.GetShape();
  size_t n = std::accumulate(
      shape.begin(), shape.end(), 1, [](const int64_t a, const int64_t b) {
        return a * b;
      });
  auto primitive_type = type.GetPrimitiveType();
  if (primitive_type == builder::PrimitiveType::S8()) {
    HLIR_CONST(int8_t);
  } else if (primitive_type == builder::PrimitiveType::S16()) {
    HLIR_CONST(int16_t);
  } else if (primitive_type == builder::PrimitiveType::S32()) {
    HLIR_CONST(int32_t);
  } else if (primitive_type == builder::PrimitiveType::S64()) {
    HLIR_CONST(int64_t);
    // has no half struct
    //   } else if (primitive_type == HlirPrimitiveType::F16()) {
    //     HLIR_CONST(half);
  } else if (primitive_type == builder::PrimitiveType::F32()) {
    HLIR_CONST(float);
  } else if (primitive_type == builder::PrimitiveType::F64()) {
    HLIR_CONST(double);
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Unable to lower scalar %d of type", primitive_type));
  }
#undef HLIR_CONST
  return hop;
}

static builder::Op ReduceAdd(builder::Op operand,
                             builder::Op init_value,
                             const std::vector<int64_t>& dimensions_to_reduce) {
  auto builder = operand.GetBuilder();
  auto dtype = operand.GetType().GetPrimitiveType();
  auto tmp_region_list = CreateBindingFunc(
      builder,
      {dtype == builder::PrimitiveType::PRED() ? BindingFuncType::OR
                                               : BindingFuncType::ADD},
      {dtype});
  std::vector<const char*> region_list;
  for (auto& region : tmp_region_list) region_list.push_back(region.c_str());
  return builder::Reduce(
      {operand}, {init_value}, dimensions_to_reduce, region_list);
}

static builder::Op SoftmaxSumOfGrad(builder::Op grad_output, int64_t dim) {
  return builder::ReduceSum(grad_output, false, {dim});
}

static builder::Op BuildSoftmaxGrad(builder::Op grad_output,
                                    builder::Op output,
                                    int64_t dim) {
  builder::Op sum = SoftmaxSumOfGrad(builder::Mul(grad_output, output), dim);
  auto broadcast_dimensions =
      GetBroadcastDimensions(grad_output.GetType().GetRank(), dim);
  sum =
      builder::BroadcastInDim(sum, broadcast_dimensions, grad_output.GetType());
  return builder::Mul(output, builder::Sub(grad_output, sum));
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SoftmaxEquivalenceTrans) {
  auto* op = node->Op();
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  builder::Op input = *(map_inputs["X"].at(0));
  if (!(input.GetType().GetPrimitiveType() == builder::PrimitiveType::F32()) &&
      !(input.GetType().GetPrimitiveType() == builder::PrimitiveType::F64())) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "GCU softmax only support FP32/FP64 datatype so far as now!"));
  }
  // to avoid 0
  double max_value_d = 1.0;
  double min_value_d = 1e-16;
  float max_value_f = 1.0;
  float min_value_f = 1e-7;
  void* max_ptr = nullptr;
  void* min_ptr = nullptr;
  auto scalar_type = builder::Type(input.GetType().GetPrimitiveType());
  if (input.GetType().GetPrimitiveType() == builder::PrimitiveType::F32()) {
    max_ptr = static_cast<void*>(&max_value_f);
    min_ptr = static_cast<void*>(&min_value_f);
  } else if (input.GetType().GetPrimitiveType() ==
             builder::PrimitiveType::F64()) {
    max_ptr = static_cast<void*>(&max_value_d);
    min_ptr = static_cast<void*>(&min_value_d);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument("UNsupport datatype"));
  }

  auto max_op = builder::Const(gcu_builder, max_ptr, scalar_type);
  auto min_op = builder::Const(gcu_builder, min_ptr, scalar_type);
  auto softmax = builder::Softmax(input, axis, true, false, 0.0);
  auto res = builder::Clamp(min_op, softmax, max_op);
  return std::make_shared<GcuOp>(res);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, SoftmaxGradEquivalenceTrans) {
  auto* op = node->Op();
  int64_t axis =
      static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));
  builder::Op dOut = *(map_inputs["Out@GRAD"].at(0));
  builder::Op out = *(map_inputs["Out"].at(0));
  if (axis < 0) axis += out.GetType().GetRank();

  builder::Op result_op = BuildSoftmaxGrad(dOut, out, axis);
  return std::make_shared<GcuOp>(result_op);
}

EQUIVALENCE_TRANS_FUNC_REG(kSoftmax, INSENSITIVE, SoftmaxEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSoftmaxGrad,
                           INSENSITIVE,
                           SoftmaxGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
