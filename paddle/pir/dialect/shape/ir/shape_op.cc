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
#include "paddle/pir/core/builtin_type.h"

namespace pir {
namespace dialect {

const char *SymbolicDim::attributes_name[attributes_num] = {"knownNegativeOne",
                                                            "knownNonNegative",
                                                            "knownNonSizeOne",
                                                            "knownNonSizeZero",
                                                            "sym_name",
                                                            "value"};  // NOLINT

void SymbolicDim::Build(
    Builder &builder,
    OperationArgument &argument,
    const std::string &sym_name,
    int64_t value,  // TODO(zhangbo) value = ShapedType::kDynamic
    bool knownNonNegative,
    bool knownNegativeOne,
    bool knownNonSizeOne,
    bool knownNonSizeZero) {
  pir::Attribute attr_sym_name =
      pir::StrAttribute::get(pir::IrContext::Instance(), sym_name);
  argument.AddAttribute("sym_name", attr_sym_name);
  pir::Attribute attr_value =
      pir::Int64Attribute::get(pir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  pir::Attribute attr_knownNonNegative =
      pir::BoolAttribute::get(pir::IrContext::Instance(), knownNonNegative);
  argument.AddAttribute("knownNonNegative", attr_knownNonNegative);
  pir::Attribute attr_knownNegativeOne =
      pir::BoolAttribute::get(pir::IrContext::Instance(), knownNegativeOne);
  argument.AddAttribute("knownNegativeOne", attr_knownNegativeOne);
  pir::Attribute attr_knownNonSizeOne =
      pir::BoolAttribute::get(pir::IrContext::Instance(), knownNonSizeOne);
  argument.AddAttribute("knownNonSizeOne", attr_knownNonSizeOne);
  pir::Attribute attr_knownNonSizeZero =
      pir::BoolAttribute::get(pir::IrContext::Instance(), knownNonSizeZero);
  argument.AddAttribute("knownNonSizeZero", attr_knownNonSizeZero);
}

const std::string SymbolicDim::getSymName() {
  return attribute<pir::StrAttribute>("sym_name").AsString();
}
int64_t SymbolicDim::getValue() {
  return attribute<pir::Int64Attribute>("value").data();
}
bool SymbolicDim::getKnownNonNegative() {
  return attribute<pir::BoolAttribute>("knownNonNegative").data();
}
bool SymbolicDim::getKnownNegativeOne() {
  return attribute<pir::BoolAttribute>("knownNegativeOne").data();
}
bool SymbolicDim::getKnownNonSizeOne() {
  return attribute<pir::BoolAttribute>("knownNonSizeOne").data();
}
bool SymbolicDim::getKnownNonSizeZero() {
  return attribute<pir::BoolAttribute>("knownNonSizeZero").data();
}

void SymbolicDim::updateSymName(std::string attrValue) {
  operation()->set_attribute(
      "sym_name",
      pir::StrAttribute::get(pir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateValue(int64_t attrValue) {
  operation()->set_attribute(
      "value", pir::Int64Attribute::get(pir::IrContext::Instance(), attrValue));
}

void SymbolicDim::updateKnownNonNegative(bool attrValue) {
  operation()->set_attribute(
      "knownNonNegative",
      pir::BoolAttribute::get(pir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNegativeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNegativeOne",
      pir::BoolAttribute::get(pir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeOne",
      pir::BoolAttribute::get(pir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeZero(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeZero",
      pir::BoolAttribute::get(pir::IrContext::Instance(), attrValue));
}

bool SymbolicDim::isDynamic() {
  return getValue() == -100000;
}  // TODO(zhangbo): getValue() == ShapedType::kDynamic;

bool SymbolicDim::merge(SymbolicDim other) {
  if (!isDynamic() && !other.isDynamic() && getValue() != other.getValue())
    return false;
  if (isDynamic() && !other.isDynamic()) updateValue(other.getValue());
  if (!isDynamic() && other.isDynamic()) other.updateValue(getValue());

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
  pir::Attribute attr_name =
      pir::StrAttribute::get(pir::IrContext::Instance(), name);
  argument.AddAttribute("name", attr_name);
  argument.output_types.emplace_back(
      pir::IndexType::get(pir::IrContext::Instance()));
}

const std::string DimOp::getName() {
  return attribute<pir::StrAttribute>("name").AsString();
}

void DimOp::setName(std::string attrName) {
  operation()->set_attribute(
      "name", pir::StrAttribute::get(pir::IrContext::Instance(), attrName));
}

const char *TieProductEqualOp::attributes_name[attributes_num] = {
    "lhs_len", "rhs_len"};  // NOLINT

void TieProductEqualOp::Build(Builder &builder,
                              OperationArgument &argument,
                              int64_t lhs_len,
                              int64_t rhs_len,
                              const std::vector<pir::OpResult> &inputs) {
  pir::Attribute attr_lhs_len =
      pir::Int64Attribute::get(pir::IrContext::Instance(), lhs_len);
  argument.AddAttribute("lhs_len", attr_lhs_len);
  pir::Attribute attr_rhs_len =
      pir::Int64Attribute::get(pir::IrContext::Instance(), rhs_len);
  argument.AddAttribute("rhs_len", attr_rhs_len);
  argument.inputs = inputs;
}

void TieProductEqualOp::Build(Builder &builder,
                              OperationArgument &argument,
                              const std::vector<pir::OpResult> &lhs,
                              const std::vector<pir::OpResult> &rhs) {
  pir::Attribute attr_lhs_len =
      pir::Int64Attribute::get(pir::IrContext::Instance(), lhs.size());
  argument.AddAttribute("lhs_len", attr_lhs_len);
  pir::Attribute attr_rhs_len =
      pir::Int64Attribute::get(pir::IrContext::Instance(), rhs.size());
  argument.AddAttribute("rhs_len", attr_rhs_len);

  argument.inputs = lhs;
  argument.inputs.insert(argument.inputs.end(), rhs.begin(), rhs.end());
}

std::vector<pir::Value> TieProductEqualOp::getLhs() {
  int64_t lhs_len = attribute<pir::Int64Attribute>("lhs_len").data();
  std::vector<pir::Value> res;
  for (uint32_t idx = 0; idx < lhs_len; idx++) {
    res.push_back(operand_source(idx));
  }
  return res;
}
std::vector<pir::Value> TieProductEqualOp::getRhs() {
  int64_t lhs_len = attribute<pir::Int64Attribute>("lhs_len").data();
  int64_t rhs_len = attribute<pir::Int64Attribute>("rhs_len").data();
  std::vector<pir::Value> res;
  for (uint32_t idx = 0; idx < rhs_len; idx++) {
    res.push_back(operand_source(lhs_len + idx));
  }
  return res;
}

const char *TieShapeOp::attributes_name[attributes_num] = {
    SymbolicDim::getSymbolicDimAttrName().c_str()};  // NOLINT

void TieShapeOp::Build(Builder &builder,
                       OperationArgument &argument,
                       const pir::OpResult &input) {
  argument.inputs = {input};
}

pir::Value TieShapeOp::getValue() { return operand_source(0); }

void FuncOp::Build(Builder &builder, OperationArgument &argument) {
  argument.num_regions = 1;
}

pir::Block *FuncOp::block() {
  pir::Region &region = (*this)->region(0);
  if (region.empty()) region.emplace_back();
  return region.front();
}

}  // namespace dialect
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::SymbolicDim)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::TieProductEqualOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::TieShapeOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::FuncOp)
