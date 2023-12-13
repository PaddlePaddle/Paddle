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
#include "paddle/pir/core/op_operand.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/program.h"

namespace pir {

std::string GetParameterNameFromValue(pir::Value value) {
  pir::Operation* owner = value.dyn_cast<OpResult>().owner();
  std::string name;
  if (owner->isa<ParameterOp>()) {
    pir::ParameterOp op = owner->dyn_cast<pir::ParameterOp>();
    name = op.param_name();
  } else if (owner->isa<ConstantTensorOp>()) {
    pir::ConstantTensorOp op = owner->dyn_cast<pir::ConstantTensorOp>();
    name = op.tensor_name();
  } else {
    PADDLE_THROW(
        phi::errors::Unimplemented("Value must be a weight from a Parameter "
                                   "or a ConstantTensorOp op."));
  }
  return name;
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
      index < op->num_operands() && op->operand_source(index),
      true,
      phi::errors::InvalidArgument("Intput operand's index must be valid."));
  return op->operand_source(index).dyn_cast<OpResult>().owner();
}

std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(Operation* op,
                                                               uint32_t index) {
  PADDLE_ENFORCE_EQ(
      index < op->num_results(),
      true,
      phi::errors::InvalidArgument("Output op result's index must be valid."));
  auto result = op->result(index);
  std::vector<std::pair<Operation*, int32_t>> use_ops;
  for (auto it = result.use_begin(); it != result.use_end(); ++it) {
    use_ops.push_back(std::make_pair(it->owner(), it->index()));
  }
  return use_ops;
}

}  // namespace pir
