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

#include <memory>
#include <string>
#include <unordered_map>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";
static constexpr char kCompilationKey[] = "compilation_key";

using LoDTensor = framework::LoDTensor;
using CinnTensor = ::cinn::hlir::framework::Tensor;
using CinnScope = ::cinn::hlir::framework::Scope;
using CinnCompiler = framework::paddle2cinn::CinnCompiler;
using CinnCompiledObject = framework::paddle2cinn::CinnCompiledObject;

namespace details {

// Tranform Paddle place to CINN target
const ::cinn::common::Target& PlaceToCinnTarget(const platform::Place& place);

// Print detailed compilation result of graph for debug
void DebugCinnCompiledResult(const CinnCompiledObject& result);

// Transform names of Paddle variables to CINN ones
std::vector<std::string> MapPaddleVariablesToCinn(
    const std::vector<std::string>& paddle_names,
    const std::unordered_map<std::string, std::string>& paddle2cinn_varmap);

// Get CinnTensor with variable name from CinnScope
std::vector<CinnTensor> GetCinnTensorsFromCompiledScope(
    const std::vector<std::string>& cinn_names, const CinnScope& cinn_scope);

// Check whether tensors from Paddle and CINN respectively
// of the same variable are equivalent in type and dimension
void CheckTensorEquivalent(const std::string& paddle_name,
                           const LoDTensor* paddle_tensor,
                           const CinnTensor& cinn_tensor);

// Allocate buffer to a Paddle tensor with assginment information from CINN
void TensorMutableDataWithCinnInfo(const platform::Place& place,
                                   const CinnTensor& cinn_tensor,
                                   LoDTensor* paddle_tensor);

// Extract temporary variable names from CinnScope by excluding
// input and output variables
std::vector<std::string> SeperateTempVar(
    const CinnScope& cinn_scope,
    const std::vector<std::string>& input_cinn_names,
    const std::vector<std::string>& output_cinn_names);

// Share the buffer of a Paddle tensor to CINN by packing memory address
// in a cinn_buffer_t object
std::unique_ptr<cinn_buffer_t> ShareTensorWithCinnBuffer(LoDTensor* tensor);

// Check all execution arguments are carried
void CheckArgumentsNotMissed(
    const CinnScope& cinn_scope,
    const std::map<std::string, cinn_pod_value_t>& name2argument);

}  // namespace details

template <typename DeviceContext, typename T>
class CinnLaunchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto& scope = ctx.scope();
    const auto& place = ctx.GetPlace();
    // Step 1. Find graph object and prepare input
    PADDLE_ENFORCE_EQ(ctx.HasAttr(kCompilationKey), true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kCompilationKey));
    const auto& compilation_key =
        ctx.template Attr<std::string>(kCompilationKey);
    VLOG(4) << "CinnLaunchOp attribute(" << kCompilationKey << ") "
            << "value:\n"
            << CinnCompiler::GetInstance()->ReadableKey(compilation_key);

    auto input_variable_names = ctx.InputNames(kX);
    const auto& input_tensors = ctx.MultiInput<LoDTensor>(kX);
    std::map<std::string, const LoDTensor*> inputs_name2tensor;
    std::transform(input_variable_names.begin(), input_variable_names.end(),
                   input_tensors.begin(),
                   std::inserter(inputs_name2tensor, inputs_name2tensor.end()),
                   [](const std::string& name, const LoDTensor* tensor) {
                     return std::make_pair(name, tensor);
                   });

    // Step 2. Get compilation result of the graph
    auto target = details::PlaceToCinnTarget(place);
    const auto& cinn_compiled_object = CinnCompiler::GetInstance()->Compile(
        compilation_key, inputs_name2tensor, target);
    details::DebugCinnCompiledResult(cinn_compiled_object);

    const auto& cinn_runtime_program = cinn_compiled_object.runtime_program;
    const auto& cinn_scope = *(cinn_compiled_object.scope);
    const auto& paddle2cinn_varmap = cinn_compiled_object.paddle2cinn_varmap;

    // Step 3. Initialize all variables needed for cinn compiled runtime
    //         program execution, and share buffers of their tensors into
    //         cinn buffers through execution arguments passed.
    VLOG(4) << "CinnLaunchOp initialize variables and prepare arguments";
    std::map<std::string, cinn_pod_value_t> name2argument;
    // because a cinn_pod_value_t does not own the cinn_buffer_t object,
    // an extra stroage is necessary to keep the object and it can
    // not be released until runtime program finish  execution.
    std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers;

    // 3.1 Prepare input variables: because tensors of input variables have
    //     been initialized before graph compiled, just check the
    //     equiality between tensors of paddle and cinn.
    auto input_cinn_names = details::MapPaddleVariablesToCinn(
        input_variable_names, paddle2cinn_varmap);
    auto input_cinn_tensors =
        details::GetCinnTensorsFromCompiledScope(input_cinn_names, cinn_scope);
    for (auto i = 0; i < input_variable_names.size(); ++i) {
      const auto& var_name = input_variable_names.at(i);
      const auto& cinn_name = input_cinn_names.at(i);
      auto* tensor = scope.GetVar(var_name)->GetMutable<LoDTensor>();
      details::CheckTensorEquivalent(var_name, tensor,
                                     input_cinn_tensors.at(i));

      VLOG(4) << "Prepare input argument-" << i << ":"
              << "name(" << var_name << "->" << cinn_name << "), "
              << "tensor(type:" << tensor->type() << ","
              << "dims:" << tensor->dims() << ").";
      auto buffer = details::ShareTensorWithCinnBuffer(tensor);
      name2argument.emplace(input_cinn_names.at(i), buffer.get());
      hold_buffers.emplace_back(std::move(buffer));
    }

    // 3.2 Prepare output variables: all output variables should
    //     be initialized and allocated buffer in advance before
    //     the runtime program start execution, the compilation result
    //     includes details of their buffer assginment which used by
    //     Paddle tensor allocation. For those variables allocated yet,
    //     like persistable parameters, just check the equiality between
    //     Paddle allocation and CINN buffer assginment.
    auto output_variable_names = ctx.OutputNames(kOutputs);
    auto output_cinn_names = details::MapPaddleVariablesToCinn(
        output_variable_names, paddle2cinn_varmap);
    auto output_cinn_tensors =
        details::GetCinnTensorsFromCompiledScope(output_cinn_names, cinn_scope);
    for (auto i = 0; i < output_variable_names.size(); ++i) {
      const auto& var_name = output_variable_names.at(i);
      const auto& cinn_name = output_cinn_names.at(i);
      auto* tensor = scope.GetVar(var_name)->GetMutable<LoDTensor>();
      if (tensor->IsInitialized()) {
        details::CheckTensorEquivalent(var_name, tensor,
                                       output_cinn_tensors.at(i));
      } else {
        details::TensorMutableDataWithCinnInfo(place, output_cinn_tensors.at(i),
                                               tensor);
      }

      VLOG(4) << "Prepare output argument-" << i << ":"
              << "name(" << var_name << "->" << cinn_name << "), "
              << "tensor(type:" << tensor->type() << ","
              << "dims:" << tensor->dims() << ").";
      auto buffer = details::ShareTensorWithCinnBuffer(tensor);
      name2argument.emplace(output_cinn_names.at(i), buffer.get());
      hold_buffers.emplace_back(std::move(buffer));
    }

    // 3.3 Prepare internal or temporary variables: Create a temporary
    //     scope to keep internal variables within graph or temporary
    //     variables needed by the compiled runtime program in addition.
    //     Here we directly use the names from CinnScope as Paddle variable
    //     names, because they will not be used outside the graph
    //     and should be destructed after computation finished.
    auto temp_variable_names = details::SeperateTempVar(
        cinn_scope, input_cinn_names, output_cinn_names);
    auto temp_scope = scope.NewTmpScope();
    if (!temp_variable_names.empty()) {
      auto temp_cinn_tensors = details::GetCinnTensorsFromCompiledScope(
          temp_variable_names, cinn_scope);
      for (auto i = 0; i < temp_variable_names.size(); ++i) {
        const auto& var_name = temp_variable_names.at(i);
        auto* tensor = temp_scope->Var(var_name)->GetMutable<LoDTensor>();
        details::TensorMutableDataWithCinnInfo(place, temp_cinn_tensors.at(i),
                                               tensor);

        VLOG(4) << "Prepare temporary argument-" << i << ":"
                << "name(" << var_name << "->" << var_name << "), "
                << "tensor(type:" << tensor->type() << ","
                << "dims:" << tensor->dims() << ").";
        auto buffer = details::ShareTensorWithCinnBuffer(tensor);
        name2argument.emplace(var_name, buffer.get());
        hold_buffers.emplace_back(std::move(buffer));
      }
    }

    // Step 4. Launch CINN to execute the compiled runtime program
    details::CheckArgumentsNotMissed(cinn_scope, name2argument);
    cinn_runtime_program->Execute(&name2argument);
    VLOG(4) << "CinnLaunchOp launch execution done.";
  }
};

}  // namespace operators
}  // namespace paddle
