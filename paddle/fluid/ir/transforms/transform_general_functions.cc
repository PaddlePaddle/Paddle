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

#include "paddle/fluid/ir/transforms/transform_general_functions.h"

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/program.h"

namespace ir {

ir::Parameter* GetParameterFromValue(ir::Value value) {
  ir::GetParameterOp op = value.GetDefiningOp()->dyn_cast<ir::GetParameterOp>();
  PADDLE_ENFORCE_NOT_NULL(
      op,
      phi::errors::InvalidArgument(
          "Value must be a weight from a GetParameter op."));
  ir::Program* program = op->GetParentProgram();
  std::string name = op->attributes()
                         .at(op.attributes_name[0])
                         .dyn_cast<ir::StrAttribute>()
                         .data();
  return program->GetParameter(name);
}

const phi::DDim& GetShapeFromValue(ir::Value value) {
  // TODO(dev): Support other types like DenseTensor.
  PADDLE_ENFORCE_EQ(
      value.type().isa<paddle::dialect::DenseTensorType>(),
      true,
      phi::errors::InvalidArgument("Value's type must be a DenseTensorType."));
  return value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
}

}  // namespace ir
