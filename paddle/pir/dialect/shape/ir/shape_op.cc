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

const char *SymbolicDimOp::attributes_name[attributes_num] = {
    "known_negative_one",   // value = -1
    "known_non_negative",   // value >= 0
    "known_non_size_one",   // value != 1
    "known_non_size_zero",  // value != 0
    "sym_name",
    "value"};  // NOLINT

void SymbolicDimOp::Build(Builder &builder,
                          OperationArgument &argument,
                          const std::string &sym_name,
                          int64_t value,
                          bool known_non_negative,
                          bool known_negative_one,
                          bool known_non_size_one,
                          bool known_non_size_zero) {
  IrContext *ctx = IrContext::Instance();
  auto attr_sym_name = StrAttribute::get(ctx, sym_name);
  auto attr_value = Int64Attribute::get(ctx, value);
  auto attr_known_none_negative = BoolAttribute::get(ctx, known_non_negative);
  auto attr_known_negative_one = BoolAttribute::get(ctx, known_negative_one);
  auto attr_known_non_size_one = BoolAttribute::get(ctx, known_non_size_one);
  auto attr_known_non_size_zero = BoolAttribute::get(ctx, known_non_size_zero);

  argument.AddAttribute("sym_name", attr_sym_name);
  argument.AddAttribute("value", attr_value);
  argument.AddAttribute("known_non_negative", attr_known_none_negative);
  argument.AddAttribute("known_negative_one", attr_known_negative_one);
  argument.AddAttribute("known_non_size_one", attr_known_non_size_one);
  argument.AddAttribute("known_non_size_zero", attr_known_non_size_zero);
}

const std::string SymbolicDimOp::GetSymName() const {
  return attribute<StrAttribute>("sym_name").AsString();
}

int64_t SymbolicDimOp::GetDimSize() const {
  return attribute<Int64Attribute>("value").data();
}

bool SymbolicDimOp::GetKnownNonNegative() {
  return attribute<BoolAttribute>("known_non_negative").data();
}

bool SymbolicDimOp::GetKnownNegativeOne() {
  return attribute<BoolAttribute>("known_negative_one").data();
}

bool SymbolicDimOp::GetKnownNonSizeOne() {
  return attribute<BoolAttribute>("known_non_size_one").data();
}

bool SymbolicDimOp::GetKnownNonSizeZero() {
  return attribute<BoolAttribute>("known_non_size_zero").data();
}

void SymbolicDimOp::SetSymName(const std::string &attr_value) {
  operation()->set_attribute(
      "sym_name", StrAttribute::get(IrContext::Instance(), attr_value));
}

void SymbolicDimOp::SetDimSize(int64_t attr_value) {
  operation()->set_attribute(
      "value", Int64Attribute::get(IrContext::Instance(), attr_value));
}

void SymbolicDimOp::UpdateKnownNonNegative(bool flag) {
  operation()->set_attribute("known_non_negative",
                             BoolAttribute::get(IrContext::Instance(), flag));
}

void SymbolicDimOp::UpdateKnownNegativeOne(bool flag) {
  operation()->set_attribute("known_negative_one",
                             BoolAttribute::get(IrContext::Instance(), flag));
}

void SymbolicDimOp::UpdateKnownNonSizeOne(bool flag) {
  operation()->set_attribute("known_non_size_one",
                             BoolAttribute::get(IrContext::Instance(), flag));
}

void SymbolicDimOp::UpdateKnownNonSizeZero(bool flag) {
  operation()->set_attribute("known_non_size_zero",
                             BoolAttribute::get(IrContext::Instance(), flag));
}

bool SymbolicDimOp::IsDynamic() const {
  return GetDimSize() == ShapedTypeInterface::kDynamic;
}

bool SymbolicDimOp::Merge(SymbolicDimOp other) {
  VLOG(4) << "Try to merge two SymbolicDimOp.";

  if (!IsDynamic() && !other.IsDynamic() && GetDimSize() != other.GetDimSize())
    return false;

  if (IsDynamic() && !other.IsDynamic()) SetDimSize(other.GetDimSize());
  if (!IsDynamic() && other.IsDynamic()) other.SetDimSize(GetDimSize());

  // eiter value >= 0
  bool known_non_negative_flag =
      GetKnownNonNegative() || other.GetKnownNonNegative();

  // eiter value == -1
  bool known_negative_one_flag =
      GetKnownNegativeOne() || other.GetKnownNegativeOne();

  if (known_non_negative_flag && known_negative_one_flag) return false;

  bool known_non_size_one_flag = GetKnownNonSizeOne() ||
                                 other.GetKnownNonSizeOne() ||
                                 known_negative_one_flag;

  bool known_non_size_zero_flag = GetKnownNonSizeZero() ||
                                  other.GetKnownNonSizeZero() ||
                                  known_negative_one_flag;

  UpdateKnownNonSizeZero(known_non_size_zero_flag);
  UpdateKnownNonSizeOne(known_non_size_one_flag);
  UpdateKnownNegativeOne(known_negative_one_flag);
  UpdateKnownNonNegative(known_non_negative_flag);
  return true;
}

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

const char *TieProductEqualOp::attributes_name[attributes_num] = {
    "lhs_len", "rhs_len"};  // NOLINT

void TieProductEqualOp::Build(Builder &builder,
                              OperationArgument &argument,
                              int64_t lhs_len,
                              int64_t rhs_len,
                              const std::vector<Value> &inputs) {
  Attribute attr_lhs_len = Int64Attribute::get(IrContext::Instance(), lhs_len);
  argument.AddAttribute("lhs_len", attr_lhs_len);
  Attribute attr_rhs_len = Int64Attribute::get(IrContext::Instance(), rhs_len);
  argument.AddAttribute("rhs_len", attr_rhs_len);
  argument.AddInputs(inputs);
}

void TieProductEqualOp::Build(Builder &builder,
                              OperationArgument &argument,
                              const std::vector<Value> &lhs,
                              const std::vector<Value> &rhs) {
  Attribute attr_lhs_len =
      Int64Attribute::get(IrContext::Instance(), lhs.size());
  argument.AddAttribute("lhs_len", attr_lhs_len);
  Attribute attr_rhs_len =
      Int64Attribute::get(IrContext::Instance(), rhs.size());
  argument.AddAttribute("rhs_len", attr_rhs_len);

  argument.AddInputs(lhs);
  argument.AddInputs(rhs);
}

std::vector<Value> TieProductEqualOp::lhs() {
  int64_t lhs_len = attribute<Int64Attribute>("lhs_len").data();
  std::vector<Value> res;
  for (uint32_t idx = 0; idx < lhs_len; idx++) {
    res.push_back(operand_source(idx));
  }
  return res;
}

std::vector<Value> TieProductEqualOp::rhs() {
  int64_t lhs_len = attribute<Int64Attribute>("lhs_len").data();
  int64_t rhs_len = attribute<Int64Attribute>("rhs_len").data();
  std::vector<Value> res;
  for (uint32_t idx = 0; idx < rhs_len; idx++) {
    res.push_back(operand_source(lhs_len + idx));
  }
  return res;
}

const char *TieShapeOp::attributes_name[attributes_num] = {
    SymbolicDimOp::GetSymbolicDimAttrName().c_str()};  // NOLINT

void TieShapeOp::Build(Builder &builder,
                       OperationArgument &argument,
                       Value input) {
  argument.AddInput(input);
}

void TieShapeOp::Build(Builder &builder,             // NOLINT
                       OperationArgument &argument,  // NOLINT
                       Value input,
                       const std::vector<Value> &dims) {
  argument.AddInput(input);
  argument.AddInputs(dims);
}

std::vector<Value> TieShapeOp::dims() {
  std::vector<Value> res;
  for (uint32_t i = 1; i < num_operands(); i++) {
    res.push_back(operand_source(i));
  }
  return res;
}

void FuncOp::Build(Builder &builder, OperationArgument &argument) {
  argument.AddRegion(nullptr);
}

Block *FuncOp::block() {
  Region &region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return &region.front();
}

void FuncOp::Print(IrPrinter &printer) {
  auto &os = printer.os;
  os << " shape.func () ";
  os << "{";
  for (auto &item : *block()) {
    os << "\n  ";
    printer.PrintOperation(&item);
  }
  os << "\n }";
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

IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::SymbolicDimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::TieProductEqualOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::TieShapeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::FuncOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::TensorDimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ShapeOfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::FromElementsOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ExtractOp);
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::ConstantIndexOp);
IR_DEFINE_EXPLICIT_TYPE_ID(pir::shape::IndexCastOp);
