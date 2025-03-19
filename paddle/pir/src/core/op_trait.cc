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

#include <glog/logging.h>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/op_trait.h"
#include "paddle/pir/include/core/type_utils.h"

namespace {

void VerifySameOperandsShapeTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameOperandsShapeTrait for : " << op->name();

  PADDLE_ENFORCE_GT(
      op->num_operands(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsShapeTrait requires at least 1 operands, "
          "but got %u operands.",
          op->name(),
          op->num_operands()));

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::Type> types;
  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  PADDLE_ENFORCE_EQ(
      VerifyCompatibleShapes(types),
      true,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsShapeTrait requires the same shape for "
          "all operands.",
          op->name()));
}

void VerifySameOperandsAndResultShapeTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameOperandsAndResultShapeTrait for : " << op->name();

  PADDLE_ENFORCE_GT(
      op->num_operands(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultShapeTrait requires at least 1 "
          "operands, but got %u operands.",
          op->name(),
          op->num_operands()));

  PADDLE_ENFORCE_GT(
      op->num_results(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultShapeTrait requires at least 1 "
          "results, but got %u results.",
          op->name(),
          op->num_results()));

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::Value> results = op->results();

  std::vector<pir::Type> types;

  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  std::for_each(results.begin(), results.end(), [&types](pir::Value op) {
    types.push_back(op.type());
  });

  PADDLE_ENFORCE_EQ(
      VerifyCompatibleShapes(types),
      true,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultShapeTrait requires compatible "
          "shapes for operands and results.",
          op->name()));
}

void VerifySameOperandsElementTypeTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameOperandsElementTypeTrait for : " << op->name();

  PADDLE_ENFORCE_GT(
      op->num_operands(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsElementTypeTrait requires at least 1 "
          "operands, but got %u operands.",
          op->name(),
          op->num_operands()));

  auto elementType = GetElementTypeOrSelf(op->result(0).type());
  for (auto operand : op->operands()) {
    PADDLE_ENFORCE_EQ(
        GetElementTypeOrSelf(operand.type()),
        elementType,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsElementTypeTrait requires the same "
            "element type for all operands.",
            op->name()));
  }
}

void VerifySameOperandsAndResultElementTypeTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameOperandsAndResultElementTypeTrait for : "
           << op->name();

  PADDLE_ENFORCE_GT(
      op->num_operands(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultElementTypeTrait requires at "
          "least 1 operands, but got %u operands.",
          op->name(),
          op->num_operands()));

  PADDLE_ENFORCE_GT(
      op->num_results(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultElementTypeTrait requires at "
          "least 1 results, but got %u results.",
          op->name(),
          op->num_results()));

  auto elementType = GetElementTypeOrSelf(op->result(0).type());

  // Verify result element type matches first result's element type.
  for (auto result : op->results()) {
    PADDLE_ENFORCE_EQ(
        GetElementTypeOrSelf(result.type()),
        elementType,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultElementTypeTrait requires the "
            "same element type for all operands and results.",
            op->name()));
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->operands()) {
    PADDLE_ENFORCE_EQ(
        GetElementTypeOrSelf(operand.type()),
        elementType,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultElementTypeTrait requires the "
            "same element type for all operands and results.",
            op->name()));
  }
}

void VerifySameOperandsAndResultTypeTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameOperandsAndResultTypeTrait for : " << op->name();

  PADDLE_ENFORCE_GT(
      op->num_operands(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultTypeTrait requires at least 1 "
          "operands, but got %u operands.",
          op->name(),
          op->num_operands()));

  PADDLE_ENFORCE_GT(
      op->num_results(),
      0,
      common::errors::InvalidArgument(
          "Op %s with SameOperandsAndResultTypeTrait requires at least 1 "
          "results, but got %u results.",
          op->name(),
          op->num_results()));

  auto type = op->result(0).type();
  auto elementType = GetElementTypeOrSelf(type);

  for (auto result : op->results()) {
    PADDLE_ENFORCE_EQ(
        GetElementTypeOrSelf(result.type()),
        elementType,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultTypeTrait requires the same "
            "type for all operands and results.",
            op->name()));

    PADDLE_ENFORCE_EQ(
        VerifyCompatibleShape(result.type(), type),
        true,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultTypeTrait requires the same "
            "type for all operands and results.",
            op->name()));
  }

  for (auto operand : op->operands()) {
    PADDLE_ENFORCE_EQ(
        GetElementTypeOrSelf(operand.type()),
        elementType,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultTypeTrait requires the same "
            "type for all operands and results.",
            op->name()));

    PADDLE_ENFORCE_EQ(
        VerifyCompatibleShape(operand.type(), type),
        true,
        common::errors::InvalidArgument(
            "Op %s with SameOperandsAndResultTypeTrait requires the same "
            "type for all operands and results.",
            op->name()));
  }
}

void VerifySameTypeOperandsTrait(const pir::Operation *op) {
  VLOG(10) << "Verify SameTypeOperandsTrait for : " << op->name();

  // For zero or only one operand.
  unsigned operand_nums = op->num_operands();
  if (operand_nums < 2) return;

  auto type = op->operand(0).type();

  for (auto operand : op->operands()) {
    PADDLE_ENFORCE_EQ(
        operand.type(),
        type,
        common::errors::InvalidArgument(
            "Op %s with SameTypeOperandsTrait requires all operands to have "
            "the same type.",
            op->name()));
  }
}

void VerifyOneResultTrait(const pir::Operation *op) {
  PADDLE_ENFORCE_EQ(
      op->num_results(),
      1,
      common::errors::InvalidArgument(
          "Op %s with OneResultTrait requires 1 result, but got %u results.",
          op->name(),
          op->num_results()));
}
}  // namespace

namespace pir {
void SameOperandsShapeTrait::Verify(Operation *op) {
  return VerifySameOperandsShapeTrait(op);
}

void SameOperandsAndResultShapeTrait::Verify(Operation *op) {
  return VerifySameOperandsAndResultShapeTrait(op);
}

void SameOperandsElementTypeTrait::Verify(Operation *op) {
  return VerifySameOperandsElementTypeTrait(op);
}

void SameOperandsAndResultElementTypeTrait::Verify(Operation *op) {
  return VerifySameOperandsAndResultElementTypeTrait(op);
}

void SameOperandsAndResultTypeTrait::Verify(Operation *op) {
  return VerifySameOperandsAndResultTypeTrait(op);
}
void SameTypeOperandsTrait::Verify(Operation *op) {
  return VerifySameTypeOperandsTrait(op);
}

void OneResultTrait::Verify(Operation *op) { return VerifyOneResultTrait(op); }
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameOperandsShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameOperandsElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameOperandsAndResultTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SameTypeOperandsTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::OneResultTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::SideEffectTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::ImmutableLayoutTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::BinaryElementWiseTrait)
