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
#include "paddle/pir/core/enforce.h"

namespace pir::dialect {

const char *SymbolicDim::attributes_name[attributes_num] = {
    "known_negative_one",   // value = -1
    "known_non_negative",   // value >= 0
    "known_non_size_one",   // value != 1
    "known_non_size_zero",  // value != 0
    "sym_name",
    "value"};  // NOLINT

void SymbolicDim::Build(Builder &builder,
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

const std::string SymbolicDim::GetSymName() {
  return attribute<StrAttribute>("sym_name").AsString();
}
int64_t SymbolicDim::GetDimSize() {
  return attribute<Int64Attribute>("value").data();
}
bool SymbolicDim::GetKnownNonNegative() {
  return attribute<BoolAttribute>("known_non_negative").data();
}
bool SymbolicDim::GetKnownNegativeOne() {
  return attribute<BoolAttribute>("known_negative_one").data();
}
bool SymbolicDim::GetKnownNonSizeOne() {
  return attribute<BoolAttribute>("known_non_size_one").data();
}
bool SymbolicDim::GetKnownNonSizeZero() {
  return attribute<BoolAttribute>("known_non_size_zero").data();
}

void SymbolicDim::SetSymName(const std::string &attr_value) {
  operation()->set_attribute(
      "sym_name", StrAttribute::get(IrContext::Instance(), attr_value));
}
void SymbolicDim::SetDimSize(int64_t attr_value) {
  operation()->set_attribute(
      "value", Int64Attribute::get(IrContext::Instance(), attr_value));
}

void SymbolicDim::UpdateKnownNonNegative(bool flag) {
  operation()->set_attribute("known_non_negative",
                             BoolAttribute::get(IrContext::Instance(), flag));
}
void SymbolicDim::UpdateKnownNegativeOne(bool flag) {
  operation()->set_attribute("known_negative_one",
                             BoolAttribute::get(IrContext::Instance(), flag));
}
void SymbolicDim::UpdateKnownNonSizeOne(bool flag) {
  operation()->set_attribute("known_non_size_one",
                             BoolAttribute::get(IrContext::Instance(), flag));
}
void SymbolicDim::UpdateKnownNonSizeZero(bool flag) {
  operation()->set_attribute("known_non_size_zero",
                             BoolAttribute::get(IrContext::Instance(), flag));
}

bool SymbolicDim::IsDynamic() {
  return GetDimSize() == ShapedTypeInterface::kDynamic;
}

bool SymbolicDim::Merge(SymbolicDim other) {
  VLOG(4) << "Try to merge two SymbolicDim ops.";

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

const std::string DimOp::getName() {
  return attribute<StrAttribute>("name").AsString();
}

void DimOp::setName(std::string attrName) {
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
    SymbolicDim::GetSymbolicDimAttrName().c_str()};  // NOLINT

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

Value TieShapeOp::value() { return operand_source(0); }

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
  return region.front();
}

void FuncOp::Print(IrPrinter &printer) {
  auto &os = printer.os;
  os << " shape.func () ";
  os << "{";
  for (auto item : *block()) {
    os << "\n  ";
    printer.PrintOperation(item);
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
  OpResult indexValue =
      builder
          .Build<ConstantOp>(Int64Attribute::get(IrContext::Instance(), index),
                             IndexType::get(IrContext::Instance()))
          ->result(0);
  argument.AddInputs({source, indexValue});
  argument.output_types.emplace_back(IndexType::get(IrContext::Instance()));
}

Value TensorDimOp::source() { return operand_source(0); }

Value TensorDimOp::index() { return operand_source(1); }

void ShapeOfOp::Build(Builder &builder,             // NOLINT
                      OperationArgument &argument,  // NOLINT
                      Value inputs) {
  argument.AddInput(inputs);
}

const std::string ShapeOfOp::getName() {
  return attribute<StrAttribute>("name").AsString();
}

}  // namespace pir::dialect

IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::SymbolicDim)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TieProductEqualOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TieShapeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::FuncOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TensorDimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::ShapeOfOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::FromElementsOp)
