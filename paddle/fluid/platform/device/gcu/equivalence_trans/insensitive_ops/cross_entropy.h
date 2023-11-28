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
#include "paddle/fluid/platform/device/gcu/equivalence_trans/utils.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
const char *const kCrossEntropy = "cross_entropy";
const char *const kCrossEntropyGrad = "cross_entropy_grad";
const char *const kCrossEntropy2 = "cross_entropy2";
const char *const kCrossEntropyGrad2 = "cross_entropy_grad2";

static builder::Op OneHot_Func(builder::Op indices,
                               int64_t depth,
                               builder::Op on_value,
                               builder::Op off_value,
                               int64_t axis) {
  auto builder = indices.GetBuilder();
  auto index_type = indices.GetType();
  auto index_shape = index_type.GetShape();
  auto index_dtype = index_type.GetPrimitiveType();
  auto dtype = on_value.GetType().GetPrimitiveType();
  int64_t dim = static_cast<int64_t>(index_shape.size());
  if (axis < 0) {
    axis += (dim + 1);
  }
  std::vector<int64_t> output_shape(index_shape);
  output_shape.insert(output_shape.begin() + axis, depth);
  auto output_index_type = builder::Type(output_shape, index_dtype);
  auto output_type = builder::Type(output_shape, dtype);
  auto iota = builder::Iota(builder, axis, output_index_type);
  std::vector<int64_t> broadcast_dims(output_shape.size());
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  broadcast_dims.erase(broadcast_dims.begin() + axis);
  auto broadcast_indices =
      builder::BroadcastInDim(indices, broadcast_dims, output_index_type);
  auto pred = builder::Compare(broadcast_indices, iota, "EQ");
  auto ons = builder::BroadcastInDim(on_value, {}, output_type);
  auto offs = builder::BroadcastInDim(off_value, {}, output_type);
  auto res = builder::Select(pred, ons, offs, output_type);
  return res;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, CrossEntropyEquivalenceTrans) {
  auto *op = node->Op();
  builder::Op inputs = *(map_inputs["X"].at(0));
  builder::Op label_op = *(map_inputs["Label"].at(0));
  auto soft_label = PADDLE_GET_CONST(bool, op->GetAttr("soft_label"));
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));

  builder::Op logits;
  builder::Op labels;
  builder::Op losses;
  if (soft_label) {
    auto input_shape = inputs.GetType().GetShape();
    auto label_shape = label_op.GetType().GetShape();
    if (input_shape.size() != label_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels Shape and Logits Shape mismatch, Expected "
          "input_dims=label_dims"));
    }
    auto input_type = inputs.GetType().GetPrimitiveType();
    auto label_type = label_op.GetType().GetPrimitiveType();
    if (input_type != label_type) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels type and Logits type mismatch"));
    }
    logits = builder::Log(inputs);
    if (ignore_index > 0) {
      auto ignore_value = builder::FullLike(label_op, ignore_index);
      auto mask = builder::Equal(ignore_value, label_op);
      // Need Convert(mask, label_op.GetType().GetPrimitiveType()) ???
      labels = label_op * mask;
    } else {
      labels = label_op;
    }
    auto loss = (-labels) * logits;
    losses = builder::ReduceSum(loss, true, {1});
    return std::make_shared<GcuOp>(losses);
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("GCU not support, It's coming soon!!"));
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               CrossEntropyGradEquivalenceTrans) {
  builder::Op x = *(map_inputs["X"].at(0));
  builder::Op label = *(map_inputs["Label"].at(0));
  builder::Op grad = *(map_inputs["Y@GRAD"].at(0));
  auto *op = node->Op();
  auto soft_label = PADDLE_GET_CONST(bool, op->GetAttr("soft_label"));
  if (soft_label) {
    auto label_shape = label.GetType().GetShape();
    auto grad_shape = grad.GetType().GetShape();
    auto x_shape = x.GetType().GetShape();
    if (label_shape.size() != grad_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels Shape and Grad Shape mismatch!"));
    }
    auto result = (-label) / x;
    result = result * grad;
    return std::make_shared<GcuOp>(result);
  } else {
    PADDLE_THROW(
        platform::errors::Unimplemented("GCU not support, It's coming soon!!"));
  }
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               CrossEntropy2EquivalenceTrans) {
  auto *op = node->Op();
  builder::Op inputs = *(map_inputs["X"].at(0));
  builder::Op label_op = *(map_inputs["Label"].at(0));
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));

  auto input_shape = inputs.GetType().GetShape();
  auto label_shape = label_op.GetType().GetShape();
  if ((input_shape.size() - 1) != label_shape.size() &&
      input_shape.size() != label_shape.size()) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Labels Shape and Logits Shape mismatch, Expected "
        "input_dims=label_dims or input_dims-1=label_dims"));
  }
  int64_t dims = static_cast<int64_t>(input_shape.size());
  for (int i = 0; i < (dims - 1); ++i) {
    if (label_shape[i] != input_shape[i]) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Logits and Labels should in same shape in dimensions except axis."
          "Logits and Label dim mismatch at %d",
          i));
    }
    if (input_shape.size() == label_shape.size()) {
      assert(label_shape[dims - 1] == 1);
    }
  }

  builder::Op labels;
  builder::Op match_x;
  builder::Op losses;
  // one-hot
  int64_t cls_index = static_cast<int64_t>(input_shape.size() - 1);
  int64_t depth = input_shape[cls_index];
  auto one_value = builder::Const(
      gcu_builder, 1, builder::Type(builder::PrimitiveType::S64()));
  auto off_value = builder::Const(
      gcu_builder, 0, builder::Type(builder::PrimitiveType::S64()));
  int64_t last_idx = static_cast<int64_t>(label_shape.size() - 1);
  int64_t label_dims = static_cast<int64_t>(label_shape.size());
  if (label_shape[last_idx] == 1) {
    std::vector<int64_t> tmp_shape;
    for (int64_t i = 0; i < (label_dims - 1); i++) {
      tmp_shape.emplace_back(label_shape[i]);
    }
    label_op = builder::Reshape(label_op, tmp_shape);
  }
  labels = OneHot_Func(label_op, depth, one_value, off_value, -1);
  labels = builder::Convert(labels,
                            builder::Type(labels.GetType().GetShape(),
                                          builder::PrimitiveType::F32()));
  auto match_op = inputs * labels;
  match_op = builder::ReduceSum(match_op, true, {1});
  auto loss = -builder::Log(match_op);
  if (ignore_index < 0) {
    match_x = match_op;
    if (input_shape.size() == label_shape.size()) {
      losses = loss;
    } else {
      losses = builder::Reshape(loss, label_shape);
    }
  } else {
    auto ignore_value =
        builder::Const(gcu_builder,
                       ignore_index,
                       builder::Type(builder::PrimitiveType::S64()));
    auto mask = builder::Equal(ignore_value, label_op, label_shape);
    match_x = match_op * mask;
    losses = mask * builder::ReduceSum(loss, true, {1});
  }
  std::vector<int64_t> tmp_shape(input_shape.begin(), input_shape.end());
  tmp_shape.push_back(0);
  auto shape_op = builder::Const(
      inputs.GetBuilder(),
      nullptr,
      builder::Type(tmp_shape, inputs.GetType().GetPrimitiveType()));
  auto tuple_result = builder::Tuple({losses, shape_op, match_x});
  auto output_name_map = op->Outputs();
  std::string output_names_attr = output_name_map["Y"][0] + ";" +
                                  output_name_map["XShape"][0] + ";" +
                                  output_name_map["MatchX"][0];
  tuple_result.SetAttribute(kAttrOpOutVarName,
                            builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(tuple_result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               CrossEntropyGrad2EquivalenceTrans) {
  builder::Op match_x = *(map_inputs["MatchX"].at(0));
  builder::Op label = *(map_inputs["Label"].at(0));
  builder::Op grad = *(map_inputs["Y@GRAD"].at(0));
  builder::Op input_shape = *(map_inputs["XShape"].at(0));
  auto *op = node->Op();
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));

  auto label_shape = label.GetType().GetShape();
  auto grad_shape = grad.GetType().GetShape();
  std::vector<int64_t> src_shape = input_shape.GetType().GetShape();
  std::vector<int64_t> tmp_shape(src_shape.begin(), src_shape.end() - 1);
  if (grad_shape.size() == (src_shape.size() - 2)) {
    grad = builder::Reshape(grad, match_x.GetType().GetShape());
  }
  builder::Op result;
  builder::Op labels;
  if (ignore_index < 0) {
    result = -grad / match_x;
  } else {
    auto ignore_value =
        builder::Const(gcu_builder,
                       ignore_index,
                       builder::Type(builder::PrimitiveType::S64()));
    auto mask = builder::Equal(ignore_value, label, label_shape);
    result = -(grad / match_x) * mask;
  }
  int64_t feature_idx = static_cast<int64_t>(tmp_shape.size() - 1);
  auto output_type =
      builder::Type(tmp_shape, input_shape.GetType().GetPrimitiveType());
  std::vector<int64_t> broadcast_dims;
  int64_t out_dims = static_cast<int64_t>(tmp_shape.size());
  for (int64_t i = 0; i < out_dims; i++) {
    broadcast_dims.emplace_back(i);
  }
  result = builder::BroadcastInDim(result, broadcast_dims, output_type);
  // one-hot
  int64_t depth = tmp_shape[feature_idx];
  auto one_value = builder::Const(
      gcu_builder, 1, builder::Type(builder::PrimitiveType::S64()));
  auto off_value = builder::Const(
      gcu_builder, 0, builder::Type(builder::PrimitiveType::S64()));
  int64_t last_idx = static_cast<int64_t>(label_shape.size() - 1);
  int64_t label_dims = static_cast<int64_t>(label_shape.size());
  if (label_shape[last_idx] == 1) {
    std::vector<int64_t> tmp_shape;
    for (int64_t i = 0; i < (label_dims - 1); i++) {
      tmp_shape.emplace_back(label_shape[i]);
    }
    label = builder::Reshape(label, tmp_shape);
  }
  labels = OneHot_Func(label, depth, one_value, off_value, -1);
  labels = builder::Convert(labels,
                            builder::Type(labels.GetType().GetShape(),
                                          builder::PrimitiveType::F32()));
  auto res_grad = result * grad * labels;
  return std::make_shared<GcuOp>(res_grad);
}

EQUIVALENCE_TRANS_FUNC_REG(kCrossEntropy,
                           INSENSITIVE,
                           CrossEntropyEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kCrossEntropyGrad,
                           INSENSITIVE,
                           CrossEntropyGradEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kCrossEntropy2,
                           INSENSITIVE,
                           CrossEntropy2EquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kCrossEntropyGrad2,
                           INSENSITIVE,
                           CrossEntropyGrad2EquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
