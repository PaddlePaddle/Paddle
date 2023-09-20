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

namespace pir {
namespace dialect {

const char *SymbolicDim::attributes_name[attributes_num] = {"knownNegativeOne",
                                                            "knownNonNegative",
                                                            "knownNonSizeOne",
                                                            "knownNonSizeZero",
                                                            "sym_name",
                                                            "value"};  // NOLINT

void SymbolicDim::Build(Builder &builder,
                        OperationArgument &argument,
                        const std::string &sym_name,
                        int64_t value,
                        bool knownNonNegative,
                        bool knownNegativeOne,
                        bool knownNonSizeOne,
                        bool knownNonSizeZero) {
  Attribute attr_sym_name = StrAttribute::get(IrContext::Instance(), sym_name);
  argument.AddAttribute("sym_name", attr_sym_name);
  Attribute attr_value = Int64Attribute::get(IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  Attribute attr_knownNonNegative =
      BoolAttribute::get(IrContext::Instance(), knownNonNegative);
  argument.AddAttribute("knownNonNegative", attr_knownNonNegative);
  Attribute attr_knownNegativeOne =
      BoolAttribute::get(IrContext::Instance(), knownNegativeOne);
  argument.AddAttribute("knownNegativeOne", attr_knownNegativeOne);
  Attribute attr_knownNonSizeOne =
      BoolAttribute::get(IrContext::Instance(), knownNonSizeOne);
  argument.AddAttribute("knownNonSizeOne", attr_knownNonSizeOne);
  Attribute attr_knownNonSizeZero =
      BoolAttribute::get(IrContext::Instance(), knownNonSizeZero);
  argument.AddAttribute("knownNonSizeZero", attr_knownNonSizeZero);
}

const std::string SymbolicDim::getSymName() {
  return attribute<StrAttribute>("sym_name").AsString();
}
int64_t SymbolicDim::getValue() {
  return attribute<Int64Attribute>("value").data();
}
bool SymbolicDim::getKnownNonNegative() {
  return attribute<BoolAttribute>("knownNonNegative").data();
}
bool SymbolicDim::getKnownNegativeOne() {
  return attribute<BoolAttribute>("knownNegativeOne").data();
}
bool SymbolicDim::getKnownNonSizeOne() {
  return attribute<BoolAttribute>("knownNonSizeOne").data();
}
bool SymbolicDim::getKnownNonSizeZero() {
  return attribute<BoolAttribute>("knownNonSizeZero").data();
}

void SymbolicDim::updateSymName(std::string attrValue) {
  operation()->set_attribute(
      "sym_name", StrAttribute::get(IrContext::Instance(), attrValue));
}
void SymbolicDim::updateValue(int64_t attrValue) {
  operation()->set_attribute(
      "value", Int64Attribute::get(IrContext::Instance(), attrValue));
}

void SymbolicDim::updateKnownNonNegative(bool attrValue) {
  operation()->set_attribute(
      "knownNonNegative", BoolAttribute::get(IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNegativeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNegativeOne", BoolAttribute::get(IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeOne", BoolAttribute::get(IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeZero(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeZero", BoolAttribute::get(IrContext::Instance(), attrValue));
}

bool SymbolicDim::IsDynamic() {
  return getValue() == ShapedTypeInterface::kDynamic;
}

bool SymbolicDim::Merge(SymbolicDim other) {
  if (!IsDynamic() && !other.IsDynamic() && getValue() != other.getValue())
    return false;
  if (IsDynamic() && !other.IsDynamic()) updateValue(other.getValue());
  if (!IsDynamic() && other.IsDynamic()) other.updateValue(getValue());

  bool knownNonNegativeFlag =
      getKnownNonNegative() || other.getKnownNonNegative();
  bool knownNegativeOneFlag =
      getKnownNegativeOne() || other.getKnownNegativeOne();
  bool knownNonSizeOneFlag = getKnownNonSizeOne() ||
                             other.getKnownNonSizeOne() || knownNegativeOneFlag;
  bool knownNonSizeZeroFlag = getKnownNonSizeZero() ||
                              other.getKnownNonSizeZero() ||
                              knownNegativeOneFlag;

  if (knownNonNegativeFlag && knownNegativeOneFlag) return false;

  updateKnownNonSizeZero(knownNonSizeZeroFlag);
  updateKnownNonSizeOne(knownNonSizeOneFlag);
  updateKnownNegativeOne(knownNegativeOneFlag);
  updateKnownNonNegative(knownNonNegativeFlag);

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
    SymbolicDim::getSymbolicDimAttrName().c_str()};  // NOLINT

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
  argument.num_regions = 1;
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
}  // namespace dialect
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::SymbolicDim)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TieProductEqualOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TieShapeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::FuncOp)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::dialect::TensorDimOp)
