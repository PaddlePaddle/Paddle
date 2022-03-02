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

#include "paddle/fluid/operators/cinn/cinn_launch_op.h"
#include <functional>
#include <vector>
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/runtime/cinn_runtime.h"
#include "cinn/runtime/flags.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(cudnn_deterministic);

namespace paddle {
namespace operators {

namespace details {

const ::cinn::common::Target& PlaceToCinnTarget(const platform::Place& place) {
  if (platform::is_cpu_place(place)) {
    return ::cinn::common::DefaultHostTarget();
  } else if (platform::is_gpu_place(place)) {
    return ::cinn::common::DefaultNVGPUTarget();
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "CINN is not supported on current place:%s", place));
  return ::cinn::common::UnkTarget();
}

void DebugCinnCompiledResult(const CinnCompiledObject& result) {
  if (!VLOG_IS_ON(4)) {
    return;
  }
  const auto& cinn_runtime_program = result.runtime_program;
  const auto& cinn_scope = *(result.scope);
  const auto& paddle2cinn_varmap = result.paddle2cinn_varmap;

  VLOG(4) << "Compiled runtime_program instrunction size:["
          << cinn_runtime_program->size() << "]";

  std::vector<std::string> infos;
  auto cinn_var_names = cinn_scope.var_names();
  infos.reserve(cinn_var_names.size());
  std::transform(cinn_var_names.begin(), cinn_var_names.end(),
                 std::back_inserter(infos),
                 [](const auto& name_view) { return name_view.data(); });
  VLOG(4) << "Compiled scope variable names:["
          << string::join_strings(infos, ',') << "]";

  infos.clear();
  infos.reserve(paddle2cinn_varmap.size());
  std::transform(paddle2cinn_varmap.begin(), paddle2cinn_varmap.end(),
                 std::back_inserter(infos), [](const auto& paddle2cinn) {
                   return paddle2cinn.first + "->" + paddle2cinn.second;
                 });
  VLOG(4) << "Compiled paddle2cinn_varmap:[" << string::join_strings(infos, ',')
          << "]";
}

void LaunchCinnExecution(const CinnCompiledObject& compiled_obj,
                         const CinnLaunchContext& context, void* stream) {
  compiled_obj.runtime_program->Execute(&context.FinalizeArguments(), stream);
}

void SetCinnRuntimeFlags() {
  VLOG(4) << "Set FLAGS_cinn_cudnn_deterministic to "
          << FLAGS_cudnn_deterministic;
  ::cinn::runtime::SetCinnCudnnDeterministic(FLAGS_cudnn_deterministic);
}

}  // namespace details

class CinnLaunchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs(kX) || ctx->HasInputs(kNoNeedBufferX),
                   "Input", string::format_string("%s|%s", kX, kNoNeedBufferX),
                   "CinnLaunchOp");
    OP_INOUT_CHECK(ctx->HasOutputs(kOutputs), "Output", kOutputs,
                   "CinnLaunchOp");
  }

 protected:
  /* [Why use single type kernel]:
   *
   * This op is similar to a control flow op, it doses not need
   * a op kernel, but in order to make it execute under dynamic
   * graph mode, implement it with op kernel.
   *
   * So whether the kernel data type is int, float or other type,
   * which has no effect on its execution logic, so directly
   * specified a data type here.
   *
   * Of course, the data type here is also not important.
   */

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

class CinnLaunchOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "(vector<LoDTensor>)"
             "which are the input of graph inside the CinnLaunchOp"
             "excluding kNoNeedBufferX.")
        .AsDuplicable();
    AddInput(kNoNeedBufferX,
             "(vector<LoDTensor>)"
             "which are the input of graph inside the CinnLaunchOp but"
             "their buffer are not needed.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput(kOutputs,
              "(vector<LoDTensor>)"
              "which are the output of graph inside the CinnLaunchOp.")
        .AsDuplicable();
    AddAttr<std::string>(
        kCompilationKey,
        "(string)"
        "a hash key used to get the graph object or its computation result.");
    AddComment(R"DOC(
CinnLaunch Operator.

This operator is used to launch CINN(https://github.com/PaddlePaddle/CINN/blob/develop/README.md)
to compile a graph and execute the compiled object.

Both input and output of this operator are a set of variables
which are input and output of the graph respectively that will be
compiled and executed in this operator.
In addition, there is an attribute named 'compilation_key' should be
set necessarily to get corresponding ir::Graph object of the graph
or its computation result.

It accomplishes the computation of graph following several steps:
  1. Fetch ir::Graph object from CinnCompiler using kCompilationKey
  2. Compile the graph to a compiled object, and insert it to the
     global cache so that we can directly query it from this cache next time
     when shape of input variables are not changed at all.
  3. Create and instantiate all variables used to execute compiled runtime program
     if necessary according to the info(type,shape) included in the return scope.
  4. Pack each tensor buffer of all above variables as execution arguments.
  5. Launch execution of the runtime program with above arguments, then
     the result would be output by writing value on underlying buffer address.

)DOC");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(CinnLaunchOpNoBufVarsInferer,
                                    kNoNeedBufferX);

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cinn_launch, ops::CinnLaunchOp, ops::CinnLaunchOpMaker,
    ops::CinnLaunchOpNoBufVarsInferer,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
/* see [Why use single type kernel] */
REGISTER_OP_CPU_KERNEL(
    cinn_launch,
    ops::CinnLaunchOpKernel<paddle::platform::CPUDeviceContext, float>);
