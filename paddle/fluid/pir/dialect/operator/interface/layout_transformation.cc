// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/pir/dialect/operator/interface/layout_transformation.h"

#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle {
namespace dialect {

template <>
common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op) {
  return common::DataLayout::NHWC;
}

template <>
common::DataLayout CurrentLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }
  return common::StringToDataLayout(data_format_attr.AsString());
}

template <>
void RewriteByLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op,
                                              common::DataLayout new_layout) {
  return;
}

template <>
void RewriteByLayoutImpl<GroupNormOp>(pir::Operation* op,
                                      common::DataLayout new_layout) {
  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<GroupNormOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<GroupNormOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<GroupNormOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<GroupNormOp>();
  return {concrete_op.y()};
}

template <>
void RewriteByLayoutImpl<ReshapeOp>(pir::Operation* op,
                                    common::DataLayout new_layout) {
  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<ReshapeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<ReshapeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  return {concrete_op.out()};
}

template <>
void RewriteByLayoutImpl<SqueezeOp>(pir::Operation* op,
                                    common::DataLayout new_layout) {
  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<SqueezeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<SqueezeOp>();
  return {concrete_op.x()};
}

template <>
std::vector<pir::Value> RelevantOutputsImpl<SqueezeOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<SqueezeOp>();
  return {concrete_op.out()};
}

}  // namespace dialect
}  // namespace paddle
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LayoutTransformationInterface)
