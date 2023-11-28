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

const char* const kSoftmaxWithCrossEntropy = "softmax_with_cross_entropy";
const char* const kSoftmaxWithCrossEntropyGrad =
    "softmax_with_cross_entropy_grad";

static builder::Op OneHot_Cus(builder::Op indices,
                              int64_t depth,
                              builder::Op on_value,
                              builder::Op off_value,
                              int64_t axis);

static builder::Op LogitsWithClamp(GcuBuilderPtr gcu_builder,
                                   builder::Op logits) {
  if (!(logits.GetType().GetPrimitiveType() == builder::PrimitiveType::F32())) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "GCU support FP32 datatype so far as now!"));
  }
  // to avoid 0
  float min_value = -64;
  auto min_op = builder::FullLike(logits, min_value);
  return builder::Clamp(min_op, logits, logits);
}

static builder::Op LogWithClamp(GcuBuilderPtr gcu_builder, builder::Op logits) {
  if (!(logits.GetType().GetPrimitiveType() == builder::PrimitiveType::F32())) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "GCU support FP32 datatype so far as now!"));
  }
  float max_value = 1e20;
  float min_value = -1e20;
  auto max_op = builder::FullLike(logits, max_value);
  auto min_op = builder::FullLike(logits, min_value);
  auto result = builder::Log(logits);
  return builder::Clamp(min_op, result, max_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               SoftmaxWithCrossEntropyEquivalenceTrans) {
  auto* op = node->Op();
  assert(op != nullptr);
  builder::Op logits_op = *(map_inputs["Logits"].at(0));
  builder::Op label_op = *(map_inputs["Label"].at(0));

  auto soft_label = PADDLE_GET_CONST(bool, op->GetAttr("soft_label"));
  auto use_softmax = PADDLE_GET_CONST(bool, op->GetAttr("use_softmax"));
  auto numeric_stable_mode =
      PADDLE_GET_CONST(bool, op->GetAttr("numeric_stable_mode"));
  auto ignore_index = PADDLE_GET_CONST(int, op->GetAttr("ignore_index"));
  auto axis = static_cast<int64_t>(PADDLE_GET_CONST(int, op->GetAttr("axis")));

  auto logits_type = logits_op.GetType().GetPrimitiveType();
  auto logits_shape = logits_op.GetType().GetShape();

  auto label_type = label_op.GetType().GetPrimitiveType();
  auto label_shape = label_op.GetType().GetShape();
  assert((axis >= -1) && (axis <= static_cast<int>(logits_shape.size()) - 1));
  if (axis < 0) {
    axis = axis + logits_shape.size();
  }

  builder::Op logits;
  builder::Op labels;
  builder::Op losses;
  builder::Op soft_logits;
  builder::Type scalar_label_type(label_shape, label_type);
  // soft_label
  if (soft_label) {
    if (logits_shape.size() != label_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels Shape and Logits Shape mismatch, Expected "
          "input_dims=label_dims"));
    }
    if (logits_type != label_type) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels type and Logits type mismatch"));
    }
    if (use_softmax) {
      logits_op = LogitsWithClamp(gcu_builder, logits_op);
      soft_logits = builder::Softmax(logits_op, axis, true, false, 0.0);
      logits = builder::Softmax(logits_op, axis, true, true, 0.0);
    } else {
      soft_logits = logits_op;
      logits = LogWithClamp(gcu_builder, soft_logits);
    }
    labels = label_op;
    // hard lable
  } else {
    if (logits_shape.size() != label_shape.size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Labels Shape and Logits Shape mismatch,Expected "
          "(input_dims==label_dims)"));
    }
    int64_t dims = static_cast<int64_t>(label_shape.size());
    for (int i = 0; i < (dims - 1); ++i) {
      if (label_shape[i] != logits_shape[i]) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Logits and Labels should in same shape in dimensions except axis."
            "Logits and Label dim mismatch at %d",
            i));
      }
    }
    if (use_softmax) {
      if (numeric_stable_mode) {
        auto max_logits = builder::ReduceMax(logits_op, true, {axis});
        auto logits_sub = logits_op - max_logits;
        logits_sub = LogitsWithClamp(gcu_builder, logits_sub);
        soft_logits = builder::Softmax(logits_sub, axis, true, false, 0.0);
        logits = builder::Softmax(logits_sub, axis, true, true, 0.0);
      } else {
        if (axis != (dims - 1)) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Attr(axis) can only be -1 "
              "when not in numeric_stable_mode."));
        }
        logits_op = LogitsWithClamp(gcu_builder, logits_op);
        soft_logits = builder::Softmax(logits_op, axis, true, false, 0.0);
        logits = builder::Softmax(logits_op, axis, true, true, 0.0);
      }
    } else {
      soft_logits = logits_op;
      logits = LogWithClamp(gcu_builder, soft_logits);
    }
    if (ignore_index > 0) {
      auto ignore_value = builder::FullLike(label_op, ignore_index);
      auto mask = builder::Equal(ignore_value, label_op);
      mask = builder::Convert(mask, scalar_label_type);
      labels = label_op * mask;
    }
    // one-hot
    int64_t cls_index = static_cast<int64_t>(logits_shape.size() - 1);
    int64_t depth = logits_shape[cls_index];
    auto one_value = builder::Const(
        gcu_builder, 1, builder::Type(builder::PrimitiveType::S32()));
    auto off_value = builder::Const(
        gcu_builder, 0, builder::Type(builder::PrimitiveType::S32()));
    int64_t last_idx = static_cast<int64_t>(label_shape.size() - 1);
    int64_t label_dims = static_cast<int64_t>(label_shape.size());
    if (label_shape[last_idx] == 1) {
      std::vector<int64_t> tmp_shape;
      for (int64_t i = 0; i < (label_dims - 1); i++) {
        tmp_shape.emplace_back(label_shape[i]);
      }
      label_op = builder::Reshape(label_op, tmp_shape);
    }
    labels = OneHot_Cus(label_op, depth, one_value, off_value, -1);
    auto label_tmp = labels.GetType().GetShape();
    labels = builder::Convert(
        labels, builder::Type(label_tmp, builder::PrimitiveType::F32()));
  }
  auto loss = (-labels) * logits;
  losses = builder::ReduceSum(loss, true, {axis});
  std::string output_names_attr;
  auto output_name_map = op->Outputs();
  output_names_attr += output_name_map["Softmax"].at(0) + ";";
  output_names_attr += output_name_map["Loss"].at(0);

  auto result_op = builder::Tuple({soft_logits, losses});
  result_op.SetAttribute(kAttrOpOutVarName,
                         builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result_op);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(gcu_builder,
                               node,
                               map_inputs,
                               running_mode,
                               SoftmaxWithCrossEntropyGradEquivalenceTrans) {
  builder::Op label = *(map_inputs["Label"].at(0));
  builder::Op softmax = *(map_inputs["Softmax"].at(0));
  builder::Op loss_grad = *(map_inputs["Loss@GRAD"].at(0));
  auto* op = node->Op();
  auto soft_label = PADDLE_GET_CONST(bool, op->GetAttr("soft_label"));
  auto use_softmax = PADDLE_GET_CONST(bool, op->GetAttr("use_softmax"));
  auto numeric_stable_mode =
      PADDLE_GET_CONST(bool, op->GetAttr("numeric_stable_mode"));

  auto label_shape = label.GetType().GetShape();
  auto grad_shape = loss_grad.GetType().GetShape();
  auto softmax_shape = softmax.GetType().GetShape();
  auto label_type = label.GetType().GetPrimitiveType();
  auto scalar_label_type = builder::Type(label_type);
  builder::Op grad_mid;
  builder::Op result;
  // soft_labe
  if (soft_label) {
    if (use_softmax) {
      grad_mid = softmax - label;
    } else {
      if (label_shape.size() != grad_shape.size()) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Labels Shape and Grad Shape mismatch!"));
      }
      grad_mid = (-label) / softmax;
    }
    // hard_lable
  } else {
    // one-hot
    int64_t cls_index = static_cast<int64_t>(softmax_shape.size() - 1);
    int64_t depth = softmax_shape[cls_index];
    auto one_value = builder::Const(
        gcu_builder, 1, builder::Type(builder::PrimitiveType::S32()));
    auto off_value = builder::Const(
        gcu_builder, 0, builder::Type(builder::PrimitiveType::S32()));
    int64_t last_idx = static_cast<int64_t>(label_shape.size() - 1);
    int64_t label_dims = static_cast<int64_t>(label_shape.size());
    if (label_shape[last_idx] == 1) {
      std::vector<int64_t> tmp_shape;
      for (int64_t i = 0; i < (label_dims - 1); i++) {
        tmp_shape.emplace_back(label_shape[i]);
      }
      label = builder::Reshape(label, tmp_shape);
    }
    label = OneHot_Cus(label, depth, one_value, off_value, -1);
    auto label_tmp = label.GetType().GetShape();
    label = builder::Convert(
        label, builder::Type(label_tmp, builder::PrimitiveType::F32()));
    if (use_softmax) {
      if (numeric_stable_mode) {
        grad_mid = softmax - label;
      }
    } else {
      grad_mid = (-label) / softmax;
    }
  }
  result = grad_mid * loss_grad;
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kSoftmaxWithCrossEntropy,
                           INSENSITIVE,
                           SoftmaxWithCrossEntropyEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kSoftmaxWithCrossEntropyGrad,
                           INSENSITIVE,
                           SoftmaxWithCrossEntropyGradEquivalenceTrans);

builder::Op OneHot_Cus(builder::Op indices,
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
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
