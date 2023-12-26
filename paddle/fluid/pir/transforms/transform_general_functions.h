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

#include "paddle/common/ddim.h"
#include "paddle/common/errors.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/core/parameter.h"
#include "paddle/pir/core/type.h"
#include "paddle/pir/core/value.h"

namespace pir {

/**
 * @brief Get the name of pararmeter from a value.
 *
 * @note The value must be a output of a ParameterOp or a ConstantTensorOp.
 *
 * @param const pir::Value&
 *
 * @return std::string
 */

std::string GetParameterNameFromValue(const pir::Value& value);

/**
 * @brief Get tensor's shape from a value.
 *
 * @param const pir::Value&
 *
 * @return const phi::DDim&
 */
const common::DDim& GetShapeFromValue(const pir::Value& value);

/**
 * @brief Get tensor's data type from a value.
 *
 * @param const pir::Value&
 *
 * @return pir::Type
 */
pir::Type GetDataTypeFromValue(const pir::Value& value);

/**
 * @brief Get an operation that defines the specific input of the operation.
 *
 * @param const Operation* const pointer to an operation
 * @param uint32_t index of operand of the operation
 *
 * @return Operation*
 */
Operation* GetDefiningOpForInput(const Operation* op, uint32_t index);

/**
 * @brief Get operations and the index of designative op operand (op result)
 that use the specific output of the operation.
 *
 * @param const Operation* cosnt pointer to an operation
 * @param uint32_t index of result of the operation

 * @return std::vector<std::pair<Operation*, int32_t>>
 */
std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(
    const Operation* op, uint32_t index);

/**
* @brief 获取指定op
*
* @param const Operation& const reference to an operation

* @return std::vector<Value>
*/
std::vector<Value> GetUsedExternalValue(const Operation& op);

/**
 * @brief Get operations and the index of designative op operand (op result)
 that use the specific output of the operation.
 *
 * @param Operation* pointer to an operation
 * @param uint32_t index of result of the operation

 * @return std::vector<std::pair<Operation*, int32_t>>
 */
std::vector<Value> GetUsedExternalValue(const Block& block);

}  // namespace pir
