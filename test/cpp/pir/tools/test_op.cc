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
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/enforce.h"

namespace test {

void RegionOp::Build(pir::Builder &builder, pir::OperationArgument &argument) {
  argument.num_regions = 1;
}

void BranchOp::Build(pir::Builder &builder,  // NOLINT
                     pir::OperationArgument &argument,
                     const std::vector<pir::OpResult> &target_operands,
                     pir::Block *target) {
  argument.AddInputs(target_operands.begin(), target_operands.end());
  argument.AddSuccessor(target);
}

void BranchOp::Verify() const {
  IR_ENFORCE((*this)->num_successors() == 1u,
             "successors number must equal to 1.");
  IR_ENFORCE((*this)->successor(0), "successor[0] can't be nullptr");
}

const char *Operation1::attributes_name[2] = {  // NOLINT
    "op1_attr1",
    "op1_attr2"};

void Operation1::Build(pir::Builder &builder,               // NOLINT
                       pir::OperationArgument &argument) {  // NOLINT
  std::unordered_map<std::string, pir::Attribute> attributes{
      {"op1_attr1", builder.str_attr("op1_attr2")},
      {"op1_attr2", builder.str_attr("op1_attr2")}};
  argument.AddOutput(builder.float32_type());
  argument.AddAttributes(attributes);
}
void Operation1::Verify() const {
  auto &attributes = this->attributes();
  if (attributes.count("op1_attr1") == 0 ||
      !attributes.at("op1_attr1").isa<pir::StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
  if (attributes.count("op1_attr2") == 0 ||
      !attributes.at("op1_attr2").isa<pir::StrAttribute>()) {
    throw("Type of attribute: parameter_name is not right.");
  }
}
}  // namespace test

IR_DEFINE_EXPLICIT_TYPE_ID(test::RegionOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::BranchOp)
IR_DEFINE_EXPLICIT_TYPE_ID(test::Operation1)
IR_DEFINE_EXPLICIT_TYPE_ID(test::Operation2)
