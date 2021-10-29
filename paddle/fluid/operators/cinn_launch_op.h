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
#include "paddle/fluid/operators/cinn_launch_op_helper.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";
static constexpr char kCompilationKey[] = "compilation_key";

using LoDTensor = framework::LoDTensor;
using Name2ConstTensor = std::map<std::string, const LoDTensor*>;
using CinnTensor = cinn::hlir::framework::Tensor;
using Name2CinnTensor = std::unordered_map<std::string, CinnTensor>;
using framework::paddle2cinn::CinnCompiler;

template <typename DeviceContext, typename T>
class CinnLaunchOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Step 1. Find graph object and prepare input
    PADDLE_ENFORCE_EQ(ctx.HasAttr(kCompilationKey), true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kCompilationKey));
    const auto& compilation_key =
        ctx.template Attr<std::string>(kCompilationKey);
    VLOG(2) << "CinnLaunchOp compilation_key:" << compilation_key;

    const auto& graph = CinnCompiler::GetInstance()->FindGraph(compilation_key);
    auto input_variable_names = ctx.InputNames(kX);
    Name2ConstTensor input_tensors =
        details::GetConstTensors(ctx.scope(), input_variable_names);

    // Step 2. Get compilation result of the graph
    auto target = details::PlaceToCinnTarget(ctx.GetPlace());
    const auto& cinn_compiled_object =
        CinnCompiler::GetInstance()->Compile(graph, input_tensors, target);
    VLOG(2) << "CinnLaunchOp compile graph done on " << ctx.GetPlace();

    const auto& cinn_runtime_program = cinn_compiled_object.runtime_program;
    const auto& compiled_scope = *(cinn_compiled_object.scope.get());
    const auto& paddle2cinn_varmap = cinn_compiled_object.paddle2cinn_varmap;

    // Step 3. Initialize all variables of the compilation runtime program
    //         in paddle, and pack them into execution arguments
    VLOG(2) << "CinnLaunchOp prepare execution arguments";
    std::map<std::string, cinn_pod_value_t> name2argument;
    std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers;
    // prepare input variables
    Name2CinnTensor input_compiled_tensors = details::GetCompiledTensors(
        input_variable_names, compiled_scope, paddle2cinn_varmap);
    details::CheckTensorEquivalent(input_tensors, input_compiled_tensors);
    details::AppendExecutionArguments(ctx.scope(), input_variable_names,
                                      paddle2cinn_varmap, &name2argument,
                                      &hold_buffers);

    // prepare output variables
    auto output_variable_names = ctx.OutputNames(kOutputs);
    Name2CinnTensor output_compiled_tensors = details::GetCompiledTensors(
        output_variable_names, compiled_scope, paddle2cinn_varmap);
    details::InitializeOutputVar(ctx.scope(), ctx.GetPlace(),
                                 output_compiled_tensors);
    Name2ConstTensor output_tensors =
        details::GetConstTensors(ctx.scope(), output_variable_names);
    details::CheckTensorEquivalent(output_tensors, output_compiled_tensors);
    details::AppendExecutionArguments(ctx.scope(), output_variable_names,
                                      paddle2cinn_varmap, &name2argument,
                                      &hold_buffers);

    // prepare temporary variables
    auto temp_variable_names =
        details::SeperateTempVar(compiled_scope, paddle2cinn_varmap,
                                 input_variable_names, output_variable_names);
    auto temp_scope = ctx.scope().NewTmpScope();
    if (!temp_variable_names.empty()) {
      details::InitializeTempVar(temp_variable_names, compiled_scope,
                                 ctx.GetPlace(), temp_scope.get());
      details::AppendExecutionArguments(*temp_scope, temp_variable_names,
                                        paddle2cinn_varmap, &name2argument,
                                        &hold_buffers);
    }

    // Step 4. Launch CINN to execute the compilation runtime program
    cinn_runtime_program->Execute(&name2argument);
    VLOG(2) << "CinnLaunchOp launch runtime_program execution done.";
  }
};

}  // namespace operators
}  // namespace paddle
