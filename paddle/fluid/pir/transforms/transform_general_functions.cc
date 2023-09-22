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

#include "paddle/fluid/pir/transforms/transform_general_functions.h"

#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"

namespace pir {

std::pair<std::string, pir::Parameter*> GetParameterFromValue(
    pir::Value value) {
  pir::GetParameterOp op =
      value.dyn_cast<OpResult>().owner()->dyn_cast<pir::GetParameterOp>();
  PADDLE_ENFORCE_NOT_NULL(
      op,
      phi::errors::InvalidArgument(
          "Value must be a weight from a GetParameter op."));
  pir::Program* program = op->GetParentProgram();
  PADDLE_ENFORCE_NOT_NULL(
      program, phi::errors::InvalidArgument("Program should not be null."));
  std::string name = op->attributes()
                         .at(op.attributes_name[0])
                         .dyn_cast<pir::StrAttribute>()
                         .AsString();
  pir::Parameter* param = program->GetParameter(name);
  PADDLE_ENFORCE_NOT_NULL(
      param, phi::errors::InvalidArgument("Parameter should not be null."));
  return {name, param};
}

const phi::DDim& GetShapeFromValue(pir::Value value) {
  // TODO(dev): Support other types like DenseTensor.
  PADDLE_ENFORCE_EQ(
      value.type().isa<paddle::dialect::DenseTensorType>(),
      true,
      phi::errors::InvalidArgument("Value's type must be a DenseTensorType."));
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

pir::Type GetDataTypeFromValue(pir::Value value) {
  // TODO(dev): Support other types like DenseTensor.
  PADDLE_ENFORCE_EQ(
      value.type().isa<paddle::dialect::DenseTensorType>(),
      true,
      phi::errors::InvalidArgument("Value's type must be a DenseTensorType."));
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dtype();
}

Operation* GetDefiningOpForInput(Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(
      index < op->num_operands(),
      true,
      phi::errors::InvalidArgument("Intput operand's index must be valid."));
  return op->operand_source(index).dyn_cast<OpResult>().owner();
}

Operation* GetFirstUseOperationForOutput(Operation* op, uint32_t index) {
  PADDLE_ENFORCE_EQ(
      index < op->num_results(),
      true,
      phi::errors::InvalidArgument("Output op result's index must be valid."));
  return op->result(index).first_use().owner();
}

}  // namespace pir
