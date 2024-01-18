// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pir/dialect/shape/ir/shape_op.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type.h"

namespace pir::shape {

const char *DimOp::attributes_name[attributes_num] = {"name"};  // NOLINT

void DimOp::Build(Builder &builder,
                  OperationArgument &argument,
                  const std::string &name) {
  Attribute attr_name = StrAttribute::get(IrContext::Instance(), name);
  argument.AddAttribute("name", attr_name);
  argument.output_types.emplace_back(IndexType::get(IrContext::Instance()));
}

const std::string DimOp::GetName() {
  return attribute<StrAttribute>("name").AsString();
}

void DimOp::SetName(std::string attrName) {
  operation()->set_attribute(
      "name", StrAttribute::get(IrContext::Instance(), attrName));
}

void TensorDimOp::Build(Builder &builder,
                        OperationArgument &argument,
                        Value source,
                        Value index) {
  argument.AddInputs({source, index});
  argument.output_types.emplace_back(IndexType::get(IrContext::Instance()));
}

void TensorDimOp::Build(Builder &builder,
                        OperationArgument &argument,
                        Value source,
                        int64_t index) {
  OpResult index_value =
      builder
          .Build<ConstantOp>(Int64Attribute::get(IrContext::Instance(), index),
                             IndexType::get(IrContext::Instance()))
          ->result(0);
  argument.AddInputs({source, index_value});
  argument.output_types.emplace_back(IndexType::get(IrContext::Instance()));
}

std::optional<int64_t> TensorDimOp::GetConstantIndex() {
  auto op = index().dyn_cast<OpResult>().owner();
  int64_t index =
      op->dyn_cast<ConstantOp>().value().dyn_cast<pir::Int64Attribute>().data();
  return index;
}

void ShapeOfOp::Build(Builder &builder,             // NOLINT
                      OperationArgument &argument,  // NOLINT
                      Value input) {
  argument.AddInput(input);

  IrContext *ctx = IrContext::Instance();
  Type dtype = IndexType::get(ctx);
  int64_t input_rank = input.type()
                           .dyn_cast<DenseTensorType>()
                           .dyn_cast<ShapedTypeInterface>()
                           .GetRank();
  pir::DDim dims = {input_rank};
  pir::DataLayout data_layout = pir::DataLayout::NCHW;
  pir::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  argument.output_types.emplace_back(
      DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset));
}

void FromElementsOp::Build(Builder &builder,             // NOLINT
                           OperationArgument &argument,  // NOLINT
                           const std::vector<Value> &elements) {
  argument.AddInputs(elements);

  IrContext *ctx = IrContext::Instance();
  Type dtype = IndexType::get(ctx);
  int64_t num_elements = elements.size();
  pir::DDim dims = {num_elements};
  pir::DataLayout data_layout = pir::DataLayout::NCHW;
  pir::LoD lod = {{0, 1, 2}};
  size_t offset = 0;

  argument.output_types.emplace_back(
      DenseTensorType::get(ctx, dtype, dims, data_layout, lod, offset));
}

std::vector<Value> FromElementsOp::elements() {
  std::vector<pir::Value> elements;
  for (uint32_t idx = 0; idx < num_operands(); idx++) {
    elements.push_back(operand_source(static_cast<int>(idx)));
  }
  return elements;
}

void ExtractOp::Build(Builder &builder,             // NOLINT
                      OperationArgument &argument,  // NOLINT
                      Value tensor,
                      std::vector<Value> indices) {
  argument.AddInput(tensor);
  argument.AddInputs(indices);
  auto type = tensor.type().dyn_cast<ShapedTypeInterface>().GetElementType();
  argument.output_types.emplace_back(type);
}

std::vector<Value> ExtractOp::indices() {
  std::vector<pir::Value> indices;
  for (uint32_t idx = 1; idx < num_operands(); idx++) {
    indices.push_back(operand_source(static_cast<int>(idx)));
  }
  return indices;
}

void ConstantIndexOp::Build(Builder &builder,
                            OperationArgument &argument,
                            int64_t value) {
  ConstantOp::Build(
      builder, argument, builder.index_attr(value), builder.index_type());
}

void IndexCastOp::Build(Builder &builder,             // NOLINT
                        OperationArgument &argument,  // NOLINT
                        Type out,
                        Value in) {
  argument.AddInput(in);
  argument.output_types.emplace_back(out);
}

}  // namespace pir::shape

IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::TensorDimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ShapeOfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::FromElementsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ExtractOp);
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ConstantIndexOp);
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::IndexCastOp);
