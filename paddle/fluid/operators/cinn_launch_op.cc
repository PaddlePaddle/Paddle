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

#include <memory>
#include <string>
#include <unordered_map>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/runtime/cinn_runtime.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/cinn_launch_op_helper.h"

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kOutputs[] = "Out";
static constexpr char kCompilationKey[] = "compilation_key";

class CinnLaunchOp : public framework::OperatorBase {
 public:
  CinnLaunchOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    VLOG(2) << "CinnLaunchOp RunImpl";
    // Step 1. Find subgraph object and prepare input
    PADDLE_ENFORCE_EQ(HasAttr(kCompilationKey), true,
                      platform::errors::NotFound(
                          "No Attribute(%s) found for CinnLaunchOp operator.",
                          kCompilationKey));
    const auto& compilation_key = Attr<std::string>(kCompilationKey);
    // TODO(CtfGo): updated after related interface ready, using local object
    // temporarily
    framework::ir::Graph temp_graph(framework::ProgramDesc());
    auto* subgraph = &temp_graph;
    // auto* subgraph = CinnCompiler::GetInstance()->FindGraph(compilation_key);
    PADDLE_ENFORCE_NOT_NULL(
        subgraph,
        platform::errors::NotFound(
            "Subgraph with compilation_key(%s) not found in CinnRunner.",
            compilation_key));

    // Step 2. Get compilation result of the subgraph
    // TODO(CtfGo): using local object temporarily,
    // will be replaced after related interface ready
    OP_INOUT_CHECK(HasInputs(kX), "Input", kX, "CinnLaunchOp");
    auto input_tensors = details::GetConstTensors(scope, Inputs(kX));
    // ::cinn::common::Target
    // auto* cinn_compiled_object = CinnCompiler::GetInstance()->Compile(
    // subgraph, input_tensors, target);
    // auto* cinn_runtime_program = cinn_compiled_object->runtime_program.get();
    // const auto& compiled_scope = *(cinn_compiled_object->scope.get());
    // const auto& paddle2cinn_varmap =
    // cinn_compiled_object->paddle2cinn_varmap;
    CinnScope compiled_scope;
    std::unique_ptr<CinnRuntimeProgram> cinn_runtime_program;
    std::unordered_map<std::string, std::string> paddle2cinn_varmap;

    // Step 3. Initialize all variables of the compilation runtime program
    //         in paddle, and pack them into execution arguments
    std::map<std::string, cinn_pod_value_t> name2argument;
    std::vector<std::unique_ptr<cinn_buffer_t>> hold_buffers;
    // prepare input variables
    auto input_compiled_tensors = details::GetCompiledTensors(
        Inputs(kX), compiled_scope, paddle2cinn_varmap);
    details::CheckTensorEquivalent(input_tensors, input_compiled_tensors);
    details::AppendExecutionArguments(scope, Inputs(kX), paddle2cinn_varmap,
                                      &name2argument, &hold_buffers);

    // prepare output variables
    auto output_tensors = details::GetConstTensors(scope, Outputs(kOutputs));
    auto output_compiled_tensors = details::GetCompiledTensors(
        Outputs(kOutputs), compiled_scope, paddle2cinn_varmap);
    details::InitializeOutputVar(scope, place, output_compiled_tensors);
    details::CheckTensorEquivalent(output_tensors, output_compiled_tensors);
    details::AppendExecutionArguments(scope, Outputs(kOutputs),
                                      paddle2cinn_varmap, &name2argument,
                                      &hold_buffers);

    // prepare temporary variables
    auto temp_variable_names = details::SeperateTempVar(
        compiled_scope, paddle2cinn_varmap, Inputs(kX), Outputs(kOutputs));
    auto temp_scope = scope.NewTmpScope();
    if (!temp_variable_names.empty()) {
      details::InitializeTempVar(temp_variable_names, compiled_scope, place,
                                 temp_scope.get());
      details::AppendExecutionArguments(*temp_scope, temp_variable_names,
                                        paddle2cinn_varmap, &name2argument,
                                        &hold_buffers);
    }

    // Step 4. Launch CINN to execute the compilation runtime program
    cinn_runtime_program->Execute(&name2argument);
  }
};

class CinnLaunchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "(vector<LoDTensor>)"
             "which are the input of subgraph inside the CinnLaunchOp.")
        .AsDuplicable();
    AddOutput(kOutputs,
              "(vector<LoDTensor>)"
              "which will be assigned with values generated by the "
              "subgraph and used by succeeding operators of the CinnLaunchOp")
        .AsDuplicable();
    AddAttr<std::string>(
        kCompilationKey,
        "(string)"
        "the hash key of serialized string of the subgraph, which "
        "will be deserialized to ir::Graph and use it to launch cinn");
    AddComment(R"DOC(
CinnLaunch Operator.

This operator is used to launch CINN(https://github.com/PaddlePaddle/CINN/blob/develop/README.md)
to compile a graph and execute the compiled object.

Input of this operator is a set of variables which are used by the subgraph
and output is a also a set of variables which are updated or generated by the graph,
and kSubGraphHashKey should be set necessarily to get the Graph object.

It accomplishs the computation of subgraph following several steps:
1. Fetch ir::Graph object from CinnRunner using kkSubGraphHashKey.
2. Compile the graph to a compiled object, and insert it to the
global cache so that we can directly query it from this cache next time
when shape of input variables are not all changed.
3. Create and instantiate all variables needed by the subgraph using
the info(type,shape) included in the compiled object.
4. Pack each tensor buffer of all variables needed by cinn as execution arguments.
5. Launch execution of cinn runtime program with above arguments,
the result would be output by writing value on underlying buffer address.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cinn_launch, ops::CinnLaunchOp, ops::CinnLaunchOpMaker);
