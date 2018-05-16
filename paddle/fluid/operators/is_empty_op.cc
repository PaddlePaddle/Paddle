/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/is_empty_op.h"

namespace paddle {
namespace operators {

// constexpr char kInput[] = "X";
// constexpr char kOutput[] = "Out";

class IsEmptyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of IsEmptyOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of IsEmptyOp should not be null.");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::Tensor>("X")->type()),
        platform::CPUPlace());
    return kt;
  }
/*
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    // get input
    auto *var = scope.FindVar(Input(kInput));
    PADDLE_ENFORCE_NOT_NULL(var);
    auto &tensor = var->Get<framework::LoDTensor>();
    // get output
    auto *out = scope.FindVar(Output(kOutput));
    PADDLE_ENFORCE_NOT_NULL(out);
    auto *out_tensor = out->GetMutable<framework::LoDTensor>();

    out_tensor->Resize({1});
    out_tensor->mutable_data<bool>(platform::CPUPlace())[0] =
        framework::product(tensor.dims()) == 0;
  }
*/
};

class IsEmptyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kInput, "(Tensor) Tensor which is to be checked.");
    AddOutput(kOutput, "(Tensor) a boolean Tensor that indicate empty or not.");
    AddComment(R"DOC(
IsEmpty Operator which checks whether a tensor is empty.

It will just return product(tensor.ddims()) > 0;
              )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(is_empty, ops::IsEmptyOp, ops::IsEmptyOpMaker);
REGISTER_OP_CPU_KERNEL(
    is_empty, ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::IsEmptyOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
