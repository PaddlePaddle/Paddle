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
#include "paddle/ir/core/type.h"
#include "paddle/ir/core/value.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

namespace ir {

/**
 * @brief Get the [name, parameter] pair of pararmeter from a value.
 *
 * @note The value must be a output of a GetParameterOp.
 *
 * @param ir::Value
 *
 * @return std::pair<std::string, ir::Parameter*>
 */

std::pair<std::string, ir::Parameter*> GetParameterFromValue(ir::Value value);

/**
 * @brief Get tensor's shape from a value.
 *
 * @param ir::Value
 *
 * @return const phi::DDim&
 */
const phi::DDim& GetShapeFromValue(ir::Value value);

/**
 * @brief Get tensor's data type from a value.
 *
 * @param ir::Value
 *
 * @return ir::Type
 */
ir::Type GetDataTypeFromValue(ir::Value value);

/**
 * @brief Get an operation that defines the specific input of the operation.
 *
 * @param Operation* pointer to an operation
 * @param uint32_t index of operand of the operation
 *
 * @return Operation*
 */
Operation* GetDefiningOpForInput(Operation* op, uint32_t index);

/**
 * @brief Get an operation that is the first to use the specific output of the
 * operation.
 *
 * @param Operation* pointer to an operation
 * @param uint32_t index of result of the operation

 * @return Operation*
 */
Operation* GetFirstUseOperationForOutput(Operation* op, uint32_t index);

}  // namespace ir
