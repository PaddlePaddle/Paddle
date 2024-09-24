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

#include "test/cpp/pir/tools/test_op.h"
#include "paddle/common/enforce.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/include/core/builtin_attribute.h"
namespace test {

void RegionOp::Build(pir::Builder &builder, pir::OperationArgument &argument) {
  argument.AddRegion(nullptr);
}

void BranchOp::Build(pir::Builder &builder,             // NOLINT
                     pir::OperationArgument &argument,  // NOLINT
                     const std::vector<pir::Value> &target_operands,
                     pir::Block *target) {
  argument.AddInputs(target_operands.begin(), target_operands.end());
  argument.AddSuccessor(target);
}

void BranchOp::VerifySig() const {
  PADDLE_ENFORCE_EQ(
      (*this)->num_successors(),
      1u,
      common::errors::InvalidArgument("successors number must equal to 1."));
  PADDLE_ENFORCE_NOT_NULL(
      (*this)->successor(0),
      common::errors::InvalidArgument("successor[0] can't be nullptr"));
}

const char *Operation1::attributes_name[2] = {"op1_attr1",   // NOLINT
                                              "op1_attr2"};  // NOLINT

void Operation1::Build(pir::Builder &builder,               // NOLINT
                       pir::OperationArgument &argument) {  // NOLINT
  std::unordered_map<std::string, pir::Attribute> attributes{
      {"op1_attr1", builder.str_attr("op1_attr2")},
      {"op1_attr2", builder.str_attr("op1_attr2")}};
  argument.AddOutput(builder.float32_type());
  argument.AddAttributes(attributes);
}
void Operation1::VerifySig() const {
  auto &attributes = this->attributes();
  if (attributes.count("op1_attr1") == 0 ||
      !attributes.at("op1_attr1").isa<pir::StrAttribute>()) {
    PADDLE_THROW(common::errors::Fatal(
        "Type of attribute: parameter_name is not right."));
  }
  if (attributes.count("op1_attr2") == 0 ||
      !attributes.at("op1_attr2").isa<pir::StrAttribute>()) {
    PADDLE_THROW(common::errors::Fatal(
        "Type of attribute: parameter_name is not right."));
  }
}

void TraitExampleOp::Build(pir::Builder &builder,             // NOLINT
                           pir::OperationArgument &argument,  // NOLINT
                           pir::Value l_operand,
                           pir::Value r_operand,
                           pir::Type out_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type);
}

void SameOperandsShapeTraitOp2::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand,
    pir::Type out_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type);
}

void SameOperandsAndResultShapeTraitOp2::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
}

void SameOperandsAndResultShapeTraitOp3::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand,
    pir::Type out_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type);
}

void SameOperandsElementTypeTraitOp2::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand,
    pir::Type out_type) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type);
}

void SameOperandsAndResultElementTypeTraitOp2::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
}

void SameOperandsAndResultElementTypeTraitOp3::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand,
    pir::Type out_type1,
    pir::Type out_type2) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type1);
  argument.AddOutput(out_type2);
}

void SameOperandsAndResultTypeTraitOp2::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
}

void SameOperandsAndResultTypeTraitOp3::Build(
    pir::Builder &builder,             // NOLINT
    pir::OperationArgument &argument,  // NOLINT
    pir::Value l_operand,
    pir::Value r_operand,
    pir::Type out_type1,
    pir::Type out_type2) {
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
  argument.AddOutput(out_type1);
  argument.AddOutput(out_type2);
}

}  // namespace test

IR_DEFINE_EXPLICIT_TYPE_ID(test::RegionOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::BranchOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::Operation2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::TraitExampleOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsShapeTraitOp1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsShapeTraitOp2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultShapeTraitOp1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultShapeTraitOp2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultShapeTraitOp3)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsElementTypeTraitOp1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsElementTypeTraitOp2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultElementTypeTraitOp3)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultTypeTraitOp1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultTypeTraitOp2)
IR_DEFINE_EXPLICIT_TYPE_ID(test::SameOperandsAndResultTypeTraitOp3)

namespace test1 {
const char *Operation1::attributes_name[2] = {"op1_attr1",   // NOLINT
                                              "op1_attr3"};  // NOLINT

void Operation1::Build(pir::Builder &builder,               // NOLINT
                       pir::OperationArgument &argument) {  // NOLINT
  std::unordered_map<std::string, pir::Attribute> attributes{
      {"op1_attr1", builder.str_attr("op1_attr1")},
      {"op1_attr3", builder.str_attr("op1_attr3")}};
  argument.AddOutput(builder.float32_type());
  argument.AddAttributes(attributes);
}
void Operation1::VerifySig() const {
  auto &attributes = this->attributes();
  if (attributes.count("op1_attr1") == 0 ||
      !attributes.at("op1_attr1").isa<pir::StrAttribute>()) {
    PADDLE_THROW(common::errors::Fatal(
        "Type of attribute: parameter_name is not right."));
  }
  if (attributes.count("op1_attr3") == 0 ||
      !attributes.at("op1_attr3").isa<pir::StrAttribute>()) {
    PADDLE_THROW(common::errors::Fatal(
        "Type of attribute: parameter_name is not right."));
  }
}

void Operation2::Build(pir::Builder &builder,               // NOLINT
                       pir::OperationArgument &argument) {  // NOLINT
  argument.AddOutput(builder.float32_type());
}

void Operation3::Build(pir::Builder &builder,  // NOLINT
                       pir::OperationArgument &argument,
                       pir::Value l_operand,
                       pir::Value r_operand) {  // NOLINT
  argument.AddInput(l_operand);
  argument.AddInput(r_operand);
}

void Operation4::Build(pir::Builder &builder,               // NOLINT
                       pir::OperationArgument &argument) {  // NOLINT
  argument.AddOutput(builder.float32_type());
  argument.AddOutput(builder.float32_type());
}
}  // namespace test1
IR_DEFINE_EXPLICIT_TYPE_ID(test1::Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(test1::Operation2)
IR_DEFINE_EXPLICIT_TYPE_ID(test1::Operation3)
IR_DEFINE_EXPLICIT_TYPE_ID(test1::Operation4)
