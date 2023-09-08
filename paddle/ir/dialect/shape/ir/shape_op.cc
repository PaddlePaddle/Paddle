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

#include "paddle/ir/dialect/shape/ir/shape_op.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_type.h"

namespace ir {
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
  ir::Attribute attr_sym_name =
      ir::StrAttribute::get(ir::IrContext::Instance(), sym_name);
  argument.AddAttribute("sym_name", attr_sym_name);
  ir::Attribute attr_value =
      ir::Int64Attribute::get(ir::IrContext::Instance(), value);
  argument.AddAttribute("value", attr_value);
  ir::Attribute attr_knownNonNegative =
      ir::BoolAttribute::get(ir::IrContext::Instance(), knownNonNegative);
  argument.AddAttribute("knownNonNegative", attr_knownNonNegative);
  ir::Attribute attr_knownNegativeOne =
      ir::BoolAttribute::get(ir::IrContext::Instance(), knownNegativeOne);
  argument.AddAttribute("knownNegativeOne", attr_knownNegativeOne);
  ir::Attribute attr_knownNonSizeOne =
      ir::BoolAttribute::get(ir::IrContext::Instance(), knownNonSizeOne);
  argument.AddAttribute("knownNonSizeOne", attr_knownNonSizeOne);
  ir::Attribute attr_knownNonSizeZero =
      ir::BoolAttribute::get(ir::IrContext::Instance(), knownNonSizeZero);
  argument.AddAttribute("knownNonSizeZero", attr_knownNonSizeZero);
}

const std::string SymbolicDim::getSymName() {
  return attribute<ir::StrAttribute>("sym_name").AsString();
}
int64_t SymbolicDim::getValue() {
  return attribute<ir::Int64Attribute>("value").data();
}
bool SymbolicDim::getKnownNonNegative() {
  return attribute<ir::BoolAttribute>("knownNonNegative").data();
}
bool SymbolicDim::getKnownNegativeOne() {
  return attribute<ir::BoolAttribute>("knownNegativeOne").data();
}
bool SymbolicDim::getKnownNonSizeOne() {
  return attribute<ir::BoolAttribute>("knownNonSizeOne").data();
}
bool SymbolicDim::getKnownNonSizeZero() {
  return attribute<ir::BoolAttribute>("knownNonSizeZero").data();
}

void SymbolicDim::updateSymName(std::string attrValue) {
  operation()->set_attribute(
      "sym_name", ir::StrAttribute::get(ir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateValue(int64_t attrValue) {
  operation()->set_attribute(
      "value", ir::Int64Attribute::get(ir::IrContext::Instance(), attrValue));
}

void SymbolicDim::updateKnownNonNegative(bool attrValue) {
  operation()->set_attribute(
      "knownNonNegative",
      ir::BoolAttribute::get(ir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNegativeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNegativeOne",
      ir::BoolAttribute::get(ir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeOne(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeOne",
      ir::BoolAttribute::get(ir::IrContext::Instance(), attrValue));
}
void SymbolicDim::updateKnownNonSizeZero(bool attrValue) {
  operation()->set_attribute(
      "knownNonSizeZero",
      ir::BoolAttribute::get(ir::IrContext::Instance(), attrValue));
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
  ir::Attribute attr_name =
      ir::StrAttribute::get(ir::IrContext::Instance(), name);
  argument.AddAttribute("name", attr_name);
  argument.output_types.emplace_back(
      ir::IndexType::get(ir::IrContext::Instance()));
}

const std::string DimOp::getName() {
  return attribute<ir::StrAttribute>("name").AsString();
}

void DimOp::setName(std::string attrName) {
  operation()->set_attribute(
      "name", ir::StrAttribute::get(ir::IrContext::Instance(), attrName));
}

const char *TieProductEqualOp::attributes_name[attributes_num] = {
    "lhs_len", "rhs_len"};  // NOLINT

void TieProductEqualOp::Build(Builder &builder,
                              OperationArgument &argument,
                              int64_t lhs_len,
                              int64_t rhs_len,
                              const std::vector<ir::OpResult> &inputs) {
  ir::Attribute attr_lhs_len =
      ir::Int64Attribute::get(ir::IrContext::Instance(), lhs_len);
  argument.AddAttribute("lhs_len", attr_lhs_len);
  ir::Attribute attr_rhs_len =
      ir::Int64Attribute::get(ir::IrContext::Instance(), rhs_len);
  argument.AddAttribute("rhs_len", attr_rhs_len);
  argument.inputs = inputs;
}

std::vector<ir::Value> TieProductEqualOp::getLhs() {
  int64_t lhs_len = attribute<ir::Int64Attribute>("lhs_len").data();
  std::vector<ir::Value> res;
  for (uint32_t idx = 0; idx < lhs_len; idx++) {
    res.push_back(operand_source(idx));
  }
  return res;
}
std::vector<ir::Value> TieProductEqualOp::getRhs() {
  int64_t lhs_len = attribute<ir::Int64Attribute>("lhs_len").data();
  int64_t rhs_len = attribute<ir::Int64Attribute>("rhs_len").data();
  std::vector<ir::Value> res;
  for (uint32_t idx = 0; idx < rhs_len; idx++) {
    res.push_back(operand_source(lhs_len + idx));
  }
  return res;
}

}  // namespace dialect
}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::SymbolicDim)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::DimOp)
IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::TieProductEqualOp)
