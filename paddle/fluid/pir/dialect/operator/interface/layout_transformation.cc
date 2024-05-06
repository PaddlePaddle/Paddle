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

#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/phi/common/scalar.h"
#include "paddle/pir/include/core/builtin_attribute.h"

namespace paddle {
namespace dialect {

template <>
common::DataLayout PreferLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op) {
  auto data_format_attr = op->attribute<pir::StrAttribute>("data_format");
  if (!data_format_attr) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "op (%s) should have attribute `data_format`, but got %s",
        op,
        data_format_attr));
  }

  auto concrete_op = op->dyn_cast<FusedConv2dAddActOp>();
  if (auto in = concrete_op.input()) {
    if (auto in_type = in.type()) {
      if (in_type.isa<paddle::dialect::DenseTensorType>()) {
        if (auto tensor_type =
                in_type.dyn_cast<paddle::dialect::DenseTensorType>()) {
          if (tensor_type.dtype().isa<pir::Float16Type>()) {
            return common::DataLayout::NHWC;
          }
        }
      }
    }
  }
  return common::StringToDataLayout(data_format_attr.AsString());
}

template <>
void RewriteByLayoutImpl<FusedConv2dAddActOp>(pir::Operation* op,
                                              common::DataLayout new_layout) {
  op->set_attribute(
      "data_format",
      pir::StrAttribute::get(pir::IrContext::Instance(),
                             common::DataLayoutToString(new_layout)));

  std::vector<pir::Type> new_outputs =
      paddle::dialect::FusedConv2dAddActOp::InferMeta(
          op->operands_source(),
          const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<GroupNormOp>(pir::Operation* op,
                                      common::DataLayout new_layout) {
  op->set_attribute(
      "data_format",
      pir::StrAttribute::get(pir::IrContext::Instance(),
                             common::DataLayoutToString(new_layout)));
  auto new_outputs = paddle::dialect::GroupNormOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
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
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function", op->name()));
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
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function", op->name()));
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

template <>
void RewriteByLayoutImpl<SiluOp>(pir::Operation* op,
                                 common::DataLayout new_layout) {
  auto new_outputs = paddle::dialect::SiluOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<AddOp>(pir::Operation* op,
                                common::DataLayout new_layout) {
  auto new_outputs = paddle::dialect::AddOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<CastOp>(pir::Operation* op,
                                 common::DataLayout new_layout) {
  auto new_outputs = paddle::dialect::CastOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
std::vector<pir::Value> RelevantInputsImpl<ConcatOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<ConcatOp>();
  return {concrete_op.x()};
}

template <>
void RewriteByLayoutImpl<ConcatOp>(pir::Operation* op,
                                   common::DataLayout new_layout) {
  // we must the value of concat axis, but this is an input
  // which is really hard to process.
  // here we handle the simple case like pd_op.full and throw
  // error in other cases.
  auto concrete_op = op->dyn_cast<ConcatOp>();
  auto axis = concrete_op.axis();
  if (!axis || !axis.defining_op->isa<paddle::dialect::FullOp>()) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Concat's axis must be processed when rewirte by layout."));
  }

  auto axis_op = axis.defining_op->dyn_cast<paddle::dialect::FullOp>();
  int axis_value = axis_op.attribute("value")
                       .dyn_cast<paddle::dialect::ScalarAttribute>()
                       .data()
                       .to<int>();

  // The layout of the tensor type is unreliable, since its always
  // NCHW, which is a default value. So we cannot deduct the new
  // axis by new layout, since we do not know if the layout changed.
  // So we simply assume the old layout must be NCHW, new layout must
  // be NHWC.
  PADDLE_ENFORCE_EQ(
      axis_value,
      1,
      common::errors::InvalidArgument(
          "Concat's axis was expected as 1, but got %d", axis_value));
  axis_op.set_attribute("value",
                        paddle::dialect::ScalarAttribute::get(
                            pir::IrContext::Instance(), phi::Scalar(3)));
}

template <>
void RewriteByLayoutImpl<pir::CombineOp>(pir::Operation* op,
                                         common::DataLayout new_layout) {
  PADDLE_THROW(common::errors::Unimplemented(
      "Op %s should have a specialized RewriteByLayout function", op->name()));
  return;
}

}  // namespace dialect
}  // namespace paddle
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LayoutTransformationInterface)
