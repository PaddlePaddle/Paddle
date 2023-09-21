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

#include "paddle/pir/core/op_trait.h"
#include "paddle/pir/core/type_util.h"

namespace pir {
namespace op_trait {
namespace impl {

bool VerifySameOperandsShapeTrait(Operation *op) {
  if (op->num_operands() < 1) return false;

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::Type> types;
  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  if (VerifyCompatibleShapes(types)) {
    return true;
  } else {
    VLOG(3) << op->name() << "requires the same shape for all operands";
    return false;
  }
}

bool VerifySameOperandsAndResultShapeTrait(Operation *op) {
  if (op->num_operands() < 1 || op->num_results() < 1) return false;

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::OpResult> results = op->results();

  std::vector<pir::Type> types;

  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  std::for_each(results.begin(), results.end(), [&types](pir::OpResult op) {
    types.push_back(op.type());
  });
  return VerifyCompatibleShapes(types);
}

bool VerifySameOperandsElementTypeTrait(Operation *op) {
  if (op->num_operands() < 1) return false;

  auto elementType = GetElementTypeOrSelf(op->result(0).type());
  for (auto operand : op->operands()) {
    if (GetElementTypeOrSelf(operand.type()) != elementType) {
      VLOG(3) << op->name()
              << "requires the same element type for all operands";
      return false;
    }
  }
  return true;
}

bool VerifySameOperandsAndResultElementTypeTrait(Operation *op) {
  if (op->num_operands() < 1 || op->num_results() < 1) return false;

  auto elementType = GetElementTypeOrSelf(op->result(0).type());

  // Verify result element type matches first result's element type.
  for (auto result : op->results()) {
    if (GetElementTypeOrSelf(result.type()) != elementType) {
      VLOG(3) << op->name()
              << "requires the same element type for all operands and results";
      return false;
    }
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->operands()) {
    if (GetElementTypeOrSelf(operand.type()) != elementType) {
      VLOG(3) << op->name()
              << "requires the same element type for all operands and results";
      return false;
    }
  }

  return true;
}

bool VerifySameOperandsAndResultTypeTrait(Operation *op) {
  if (op->num_operands() < 1 || op->num_results() < 1) return false;

  auto type = op->result(0).type();
  auto elementType = GetElementTypeOrSelf(type);

  for (auto result : op->results()) {
    if (GetElementTypeOrSelf(result.type()) != elementType ||
        VerifyCompatibleShape(result.type(), type)) {
      VLOG(3) << op->name()
              << "requires the same type for all operands and results";
      return false;
    }
  }

  for (auto operand : op->operands()) {
    if (GetElementTypeOrSelf(operand.type()) != elementType ||
        VerifyCompatibleShape(operand.type(), type)) {
      VLOG(3) << op->name()
              << "requires the same type for all operands and results";
      return false;
    }
  }
  return true;
}

bool VerifySameTypeOperandsTrait(Operation *op) {
  // For zero or only one operand.
  unsigned operand_nums = op->num_operands();
  if (operand_nums < 2) return true;

  auto type = op->operand(0).type();

  for (auto operand : op->operands())
    if (operand.type() != type) {
      VLOG(3) << op->name() << "requires all operands to have the same type";
      return false;
    }
  return true;
}

}  // namespace impl
}  // namespace op_trait
}  // namespace pir

IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameTypeOperandsTrait)
