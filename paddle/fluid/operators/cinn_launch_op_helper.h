// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <unordered_map>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Name2ConstTensor = std::unordered_map<std::string, const LoDTensor*>;
using Variable = framework::Variable;
using Scope = framework::Scope;

using CinnTensor = cinn::hlir::framework::Tensor;
using Name2CinnTensor = std::unordered_map<std::string, CinnTensor>;
using CinnScope = cinn::hlir::framework::Scope;
using CinnRuntimeProgram = cinn::hlir::framework::Program;

namespace details {

// Get the underlying tensor of a variable,
// result: paddle name --> const LoDTensor*
Name2ConstTensor GetConstTensors(
    const Scope& scope, const std::vector<std::string>& variable_names);

// Get the compiled tensor of a paddle variable,
// result: paddle name --> CinnTensor
Name2CinnTensor GetCompiledTensors(
    const std::vector<std::string>& paddle_var_names,
    const CinnScope& compiled_scope,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap);

// Check a original tensor of Paddle is equivalent
// to the complied tensor from CINN
void CheckTensorEquivalent(
    const Name2ConstTensor& paddle_tensors, /*paddle name -> const LoDTensor**/
    const Name2CinnTensor& compiled_tensors /*paddle name -> CinnTensor*/);

// Initialize output variables with the compilation result from CINN
void InitializeOutputVar(
    const Scope& scope, const platform::Place& place,
    const Name2CinnTensor& compiled_tensors /*paddle name -> CinnTensor*/);

// Extract extral temporary variables by
// excluding input/output variables from compiled scope
std::vector<std::string> SeperateTempVar(
    const CinnScope& compiled_scope,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    const std::vector<std::string>& input_var_names,
    const std::vector<std::string>& output_var_names);

// Initialize temporary variables in a temp scope,
// using the definition in compiled_scope
void InitializeTempVar(const std::vector<std::string>& variable_names,
                       const CinnScope& compiled_scope,
                       const platform::Place& place, Scope* temp_scope);

// Share paddle tensor to a cinn one through cinn_buffer_t object
void SharePaddleTensorWithCinnBuffer(LoDTensor* paddle_tensor,
                                     cinn_buffer_t* cinn_buffer);

// Pack tensors of all variables as execution arguments,
// which will be passed into compilation runtime program to execute
void AppendExecutionArguments(
    const Scope& scope, const std::vector<std::string>& variable_names,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap,
    std::map<std::string, cinn_pod_value_t>* name2argument,
    std::vector<std::unique_ptr<cinn_buffer_t>>* hold_buffers);

}  // namespace details

}  // namespace operators
}  // namespace paddle
