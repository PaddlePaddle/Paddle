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
#include "paddle/pir/core/enforce.h"
#include "paddle/pir/core/type_util.h"

namespace pir::op_trait::impl {

void VerifySameOperandsShapeTrait(Operation *op) {
  VLOG(4) << "Verify SameOperandsShapeTrait for : " << op->name();

  IR_ENFORCE(op->num_operands() > 0,
             "Op %s with SameOperandsShapeTrait requires at least 1 operands, "
             "but got %u operands.",
             op->name(),
             op->num_operands());

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::Type> types;
  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  IR_ENFORCE(VerifyCompatibleShapes(types),
             "Op %s with SameOperandsShapeTrait requires the same shape for "
             "all operands.",
             op->name());
}

void VerifySameOperandsAndResultShapeTrait(Operation *op) {
  VLOG(4) << "Verify SameOperandsAndResultShapeTrait for : " << op->name();

  IR_ENFORCE(op->num_operands() > 0,
             "Op %s with SameOperandsAndResultShapeTrait requires at least 1 "
             "operands, but got %u operands.",
             op->name(),
             op->num_operands());

  IR_ENFORCE(op->num_results() > 0,
             "Op %s with SameOperandsAndResultShapeTrait requires at least 1 "
             "results, but got %u results.",
             op->name(),
             op->num_results());

  std::vector<pir::OpOperand> operands = op->operands();
  std::vector<pir::OpResult> results = op->results();

  std::vector<pir::Type> types;

  std::for_each(operands.begin(), operands.end(), [&types](pir::OpOperand op) {
    types.push_back(op.type());
  });

  std::for_each(results.begin(), results.end(), [&types](pir::OpResult op) {
    types.push_back(op.type());
  });

  IR_ENFORCE(VerifyCompatibleShapes(types),
             "Op %s with SameOperandsAndResultShapeTrait requires compatible "
             "shapes for operands and results.");
}

void VerifySameOperandsElementTypeTrait(Operation *op) {
  VLOG(4) << "Verify SameOperandsElementTypeTrait for : " << op->name();

  IR_ENFORCE(op->num_operands() > 0,
             "Op %s with SameOperandsElementTypeTrait requires at least 1 "
             "operands, but got %u operands.",
             op->name(),
             op->num_operands());

  auto elementType = GetElementTypeOrSelf(op->result(0).type());
  for (auto operand : op->operands()) {
    IR_ENFORCE(GetElementTypeOrSelf(operand.type()) == elementType,
               "Op %s with SameOperandsElementTypeTrait requires the same "
               "element type for all operands.",
               op->name());
  }
}

void VerifySameOperandsAndResultElementTypeTrait(Operation *op) {
  VLOG(4) << "Verify SameOperandsAndResultElementTypeTrait for : "
          << op->name();

  IR_ENFORCE(op->num_operands() > 0,
             "Op %s with SameOperandsAndResultElementTypeTrait requires at "
             "least 1 operands, but got %u operands.",
             op->name(),
             op->num_operands());

  IR_ENFORCE(op->num_results() > 0,
             "Op %s with SameOperandsAndResultElementTypeTrait requires at "
             "least 1 results, but got %u results.",
             op->name(),
             op->num_results());

  auto elementType = GetElementTypeOrSelf(op->result(0).type());

  // Verify result element type matches first result's element type.
  for (auto result : op->results()) {
    IR_ENFORCE(GetElementTypeOrSelf(result.type()) == elementType,
               "Op %s with SameOperandsAndResultElementTypeTrait requires the "
               "same element type for all operands and results.",
               op->name());
  }

  // Verify operand's element type matches first result's element type.
  for (auto operand : op->operands()) {
    IR_ENFORCE(GetElementTypeOrSelf(operand.type()) == elementType,
               "Op %s with SameOperandsAndResultElementTypeTrait requires the "
               "same element type for all operands and results.",
               op->name());
  }
}

void VerifySameOperandsAndResultTypeTrait(Operation *op) {
  VLOG(4) << "Verify SameOperandsAndResultTypeTrait for : " << op->name();

  IR_ENFORCE(op->num_operands() > 0,
             "Op %s with SameOperandsAndResultTypeTrait requires at least 1 "
             "operands, but got %u operands.",
             op->name(),
             op->num_operands());

  IR_ENFORCE(op->num_results() > 0,
             "Op %s with SameOperandsAndResultTypeTrait requires at least 1 "
             "results, but got %u results.",
             op->name(),
             op->num_results());

  auto type = op->result(0).type();
  auto elementType = GetElementTypeOrSelf(type);

  for (auto result : op->results()) {
    IR_ENFORCE(GetElementTypeOrSelf(result.type()) == elementType,
               "Op %s with SameOperandsAndResultTypeTrait requires the same "
               "type for all operands and results.",
               op->name());

    IR_ENFORCE(VerifyCompatibleShape(result.type(), type),
               "Op %s with SameOperandsAndResultTypeTrait requires the same "
               "type for all operands and results.",
               op->name());
  }

  for (auto operand : op->operands()) {
    IR_ENFORCE(GetElementTypeOrSelf(operand.type()) == elementType,
               "Op %s with SameOperandsAndResultTypeTrait requires the same "
               "type for all operands and results.",
               op->name());

    IR_ENFORCE(VerifyCompatibleShape(operand.type(), type),
               "Op %s with SameOperandsAndResultTypeTrait requires the same "
               "type for all operands and results.",
               op->name());
  }
}

void VerifySameTypeOperandsTrait(Operation *op) {
  VLOG(4) << "Verify SameTypeOperandsTrait for : " << op->name();

  // For zero or only one operand.
  unsigned operand_nums = op->num_operands();
  if (operand_nums < 2) return;

  auto type = op->operand(0).type();

  for (auto operand : op->operands()) {
    IR_ENFORCE(operand.type() == type,
               "Op %s with SameTypeOperandsTrait requires all operands to have "
               "the same type.",
               op->name());
  }
}

}  // namespace  pir::op_trait::impl

IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultShapeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultElementTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameOperandsAndResultTypeTrait)
IR_DEFINE_EXPLICIT_TYPE_ID(pir::op_trait::SameTypeOperandsTrait)
