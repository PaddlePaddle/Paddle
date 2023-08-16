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

std::string SymbolicDim::getSymName() {
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

}  // namespace dialect
}  // namespace ir

IR_DEFINE_EXPLICIT_TYPE_ID(ir::dialect::SymbolicDim)
