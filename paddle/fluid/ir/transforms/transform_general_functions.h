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

#pragma once

#include "paddle/ir/core/operation.h"
#include "paddle/ir/core/parameter.h"
#include "paddle/ir/core/value.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace ir {

/**
 * @brief Get the parameter from a value.
 *
 * @note The value must be a output of a GetParameterOp.
 *
 * @param ir::Value
 *
 * @return ir::Parameter*
 */
ir::Parameter* GetParameterFromValue(ir::Value value);

/**
 * @brief Get tensor's shape from a value.
 *
 * @param ir::Value
 *
 * @return const phi::DDim&
 */
const phi::DDim& GetShapeFromValue(ir::Value value);

/**
 * @brief Get an operation that defines the specific input of the operation.
 *
 * @param Operation*
 *
 * @return Operation*
 */
template <uint32_t Index = 0>
Operation* GetDefiningOpForInput(Operation* op) {
  PADDLE_ENFORCE_EQ(
      Index < op->num_operands(),
      true,
      phi::errors::InvalidArgument("Intput operand's index must be valid."));
  return op->operand(Index).GetDefiningOp();
}

/**
 * @brief Get an operation that is the first to use the specific output of the
 * operation.
 *
 * @param Operation*
 *
 * @return Operation*
 */
template <uint32_t Index = 0>
Operation* GetFirstUseOperationForOutput(Operation* op) {
  PADDLE_ENFORCE_EQ(
      Index < op->num_results(),
      true,
      phi::errors::InvalidArgument("Output op result's index must be valid."));
  return op->result(Index).first_use().owner();
}

}  // namespace ir
