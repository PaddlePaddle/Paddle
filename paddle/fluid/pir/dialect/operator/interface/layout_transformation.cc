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
#include <exception>
#include <utility>

#include "paddle/common/ddim.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
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
  auto concrete_op = op->dyn_cast<ReshapeOp>();
  auto shape_value = concrete_op.shape();
  if (!shape_value) return;
  auto shape_op = shape_value.defining_op<FullIntArrayOp>();
  if (!shape_op) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Reshape op's input `shape` is from %s instead of FullIntArrayOp",
        shape_op->name()));
  }
  // TODO(lyk): we must assert this full int array op has one user which is
  // reshape
  auto values =
      shape_op->attribute("value").dyn_cast<pir::ArrayAttribute>().AsVector();
  std::vector<int64_t> cur_values;
  for (auto v : values) {
    cur_values.push_back(v.dyn_cast<pir::Int64Attribute>().data());
  }
  PADDLE_ENFORCE_GE(
      cur_values.size(),
      4,
      phi::errors::InvalidArgument(
          "Reshape op's input `shape` should have a size > 4, but we got %d ",
          cur_values.size()));

  // there should be only a non-1 value
  int non_one_cnt = 0;
  int non_one_idx = -1;
  for (size_t i = 0; i < cur_values.size(); ++i) {
    auto v = cur_values[i];
    if (v != 1) {
      non_one_cnt++;
      non_one_idx = i;
    }
  }
  if (non_one_cnt > 1) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Reshape op's input `shape` is from %s, which has more than one non-1 "
        "value: %d",
        non_one_cnt));
  }

  // in this case we still have no way to get the real layout
  // so assume the old one is NCHW, the new one is NHWC.
  PADDLE_ENFORCE_EQ(non_one_idx,
                    1,
                    phi::errors::InvalidArgument(
                        "Reshape op's input `shape` should set `x` to NCHW, so "
                        "idx would be 1,  but we got %d ",
                        non_one_idx));

  // makes 1xCx1x1 to 1x1x1xC
  std::swap(cur_values[1], cur_values[3]);

  std::vector<pir::Attribute> cur_values_attr;
  for (auto v : cur_values) {
    cur_values_attr.push_back(pir::Int64Attribute::get(op->ir_context(), v));
  }

  auto new_attr = pir::ArrayAttribute::get(op->ir_context(), cur_values_attr);
  shape_op->set_attribute("value", new_attr);

  // infer new meta for full int array op
  auto new_outputs = paddle::dialect::FullIntArrayOp::InferMeta(
      shape_op->operands_source(),
      const_cast<pir::AttributeMap*>(&shape_op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    shape_op->result(i).set_type(new_outputs[i]);
  }

  // infer new meta for reshape op
  auto new_reshape_outputs = paddle::dialect::ReshapeOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_reshape_outputs.size(); ++i) {
    op->result(i).set_type(new_reshape_outputs[i]);
  }

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
  if (!axis || !(axis.defining_op()->isa<FullOp>())) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Concat's axis must be processed when rewirte by layout."));
  }

  // TODO(lyk): we must assert this full int array op has one user which is
  // reshape
  auto axis_op = axis.defining_op()->dyn_cast<FullOp>();
  int axis_value =
      axis_op.attribute("value").dyn_cast<ScalarAttribute>().data().to<int>();

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
  axis.defining_op()->set_attribute(
      "value",
      ScalarAttribute::get(pir::IrContext::Instance(), phi::Scalar(3)));

  // infer new meta for concat
  auto new_outputs = ConcatOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<pir::CombineOp>(pir::Operation* op,
                                         common::DataLayout new_layout) {
  auto concrete_op = op->dyn_cast<pir::CombineOp>();
  auto out = concrete_op.out();
  if (!out) return;
  std::vector<pir::Type> new_out_type;
  for (auto v : op->operands_source()) {
    new_out_type.push_back(v.type());
  }
  auto new_out_type_v =
      pir::VectorType::get(pir::IrContext::Instance(), new_out_type);
  out.set_type(new_out_type_v);

  return;
}

template <>
std::vector<pir::Value> RelevantInputsImpl<Pool2dOp>(pir::Operation* op) {
  auto concrete_op = op->dyn_cast<Pool2dOp>();
  return {concrete_op.x()};
}

template <>
void RewriteByLayoutImpl<Pool2dOp>(pir::Operation* op,
                                   common::DataLayout new_layout) {
  op->set_attribute(
      "data_format",
      pir::StrAttribute::get(pir::IrContext::Instance(),
                             common::DataLayoutToString(new_layout)));

  std::vector<pir::Type> new_outputs = Pool2dOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<MultiplyOp>(pir::Operation* op,
                                     common::DataLayout new_layout) {
  std::vector<pir::Type> new_outputs = MultiplyOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<AssignOp>(pir::Operation* op,
                                   common::DataLayout new_layout) {
  std::vector<pir::Type> new_outputs = AssignOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

template <>
void RewriteByLayoutImpl<SwishOp>(pir::Operation* op,
                                  common::DataLayout new_layout) {
  std::vector<pir::Type> new_outputs = SwishOp::InferMeta(
      op->operands_source(), const_cast<pir::AttributeMap*>(&op->attributes()));
  for (size_t i = 0; i < new_outputs.size(); ++i) {
    op->result(i).set_type(new_outputs[i]);
  }
}

}  // namespace dialect
}  // namespace paddle
IR_DEFINE_EXPLICIT_TYPE_ID(paddle::dialect::LayoutTransformationInterface)
