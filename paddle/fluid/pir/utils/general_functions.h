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

#include <string>
#include <vector>

#include "paddle/pir/include/core/type.h"
#include "paddle/pir/include/pass/pass.h"

namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace phi {
class DenseTensor;
}  // namespace phi

namespace pir {

class Operation;
class Block;
class Value;
class PatternRewriter;
class Parameter;
class Type;

using Variable = paddle::framework::Variable;
using Scope = paddle::framework::Scope;

pir::Type TranslateToIrDataType(phi::DenseTensor* tensor);

Parameter* GetParameter(Operation* op, const std::string& name);

pir::Operation* CreateOpeartionByName(const std::string& op_name,
                                      const std::vector<pir::Value>& inputs,
                                      const pir::AttributeMap& attrs,
                                      const pir::PatternRewriter& rewriter);

// IsType  GetMutable
template <typename T>
T* VarGetMutable(Variable* var);

template <typename T>
bool VarIsType(Variable* var);

Variable* ScopeFindVar(Scope* scope_, const std::string& name);

Variable* ScopeGetVar(Scope* scope_, const std::string& name);

Variable* ScopeVar(Scope* scope_, const std::string& name);

std::vector<std::string> ScopeGetVarNames(Scope* scope_);

Scope* GetScopeImpl(pir::Pass* pass);

/**
 * @brief Get the name of parameter from a value.
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
 * @return std::vector<int64_t>
 */
std::vector<int64_t> GetShapeFromValue(const pir::Value& value);

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
TEST_API Operation* GetDefiningOpForInput(const Operation* op, uint32_t index);

/**
 * @brief Get operations and the index of designative op operand (op result)
 that use the specific output of the operation.
 *
 * @param const Operation* const pointer to an operation
 * @param uint32_t index of result of the operation

 * @return std::vector<std::pair<Operation*, int32_t>>
 */
std::vector<std::pair<Operation*, int32_t>> GetUseOpsForOutput(
    const Operation* op, uint32_t index);

/**
* @brief Get the value of the input and output of the specified op in the
external block.
*
* @param const Operation& const reference to an operation

* @return std::vector<Value>
*/
std::vector<Value> GetUsedExternalValue(const Operation& op);

/**
 * @brief Get the external value of the input and output of all op which in the
 specified block.
 *
 * @param const Block& const reference to an block

 * @return std::vector<Value>
 */
std::vector<Value> GetUsedExternalValue(const Block& block);

/**
 * @brief Determine whether a value comes from a weight or has no input op. That
 is to say, it is permissible.
 *
 * @param const pir::Value&

 * @return bool
 */
bool ValueIsPersistable(const pir::Value& value);

}  // namespace pir
