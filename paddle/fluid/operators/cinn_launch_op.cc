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

#include "paddle/fluid/operators/cinn_launch_op.h"

namespace paddle {
namespace operators {

class CinnLaunchOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs(kX), "Input", kX, "CinnLaunchOp");
    OP_INOUT_CHECK(ctx->HasOutput(kOutputs), "Output", kOutputs,
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
             "which are the input of graph inside the CinnLaunchOp.")
        .AsDuplicable();
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

It accomplishs the computation of graph following several steps:
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

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cinn_launch, ops::CinnLaunchOp, ops::CinnLaunchOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
/* see [Why use single type kernel] */
REGISTER_OP_CPU_KERNEL(
    cinn_launch,
    ops::CinnLaunchOpKernel<paddle::platform::CPUDeviceContext, float>);
